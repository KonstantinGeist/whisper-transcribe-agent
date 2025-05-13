package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"github.com/pkg/errors"
)

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionRequest struct {
	Messages []ChatMessage `json:"messages"`
}

func main() {
	fmt.Println("whisper-transcribe-agent - route OpenAI-compatible Chat Completions API to a whisper-server instance")

	port := flag.String("port", ":8080", "HTTP server listen address")
	baseURL := flag.String("whisper-server-url", "", "Base URL of transcription service")
	whisperModel := flag.String("whisper-model", "", "Whisper model to use")
	maxAudioSize := flag.Int64("max-audio-size", 0, "The maximum size of the audio")
	flag.Parse()

	if *baseURL == "" {
		log.Fatal("whisper-server-url is required")
	}
	if *whisperModel == "" {
		log.Fatal("whisper-model is required")
	}
	if *maxAudioSize == 0 {
		log.Fatal("max-audio-size is required")
	}

	http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		respond := func(text string, err error) {
			if err != nil {
				text = fmt.Sprintf("%s: %s", text, err.Error())
			}

			response := map[string]interface{}{
				"id":      "chatcmpl-mockid",
				"object":  "chat.completion",
				"created": time.Now().Unix(),
				"model":   *whisperModel,
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"message": map[string]string{
							"role":    "assistant",
							"content": text,
						},
						"finish_reason": "stop",
					},
				},
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)

			fmt.Printf("responded with: %s\n", text)
			if err != nil {
				fmt.Printf("stacktrace: %+v\n", err)
			}
		}

		if r.Method != http.MethodPost {
			respond("Method not allowed", nil)
			return
		}

		var chatReq ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&chatReq); err != nil {
			respond("Invalid JSON", errors.WithStack(err))
			return
		}

		if len(chatReq.Messages) == 0 {
			respond("No messages provided", nil)
			return
		}

		lastMsg := chatReq.Messages[len(chatReq.Messages)-1]
		audioURL := extractURLFromText(lastMsg.Content)
		if audioURL == "" {
			respond("No audio URL found in message", nil)
			return
		}

		fmt.Printf("new request for file: %s\n", audioURL)

		audioData, err := downloadFileWithLimit(audioURL, *maxAudioSize)
		if err != nil {
			respond("Failed to download audio", errors.WithStack(err))
			return
		}

		respBody, _, err := sendToTranscription(*baseURL, *whisperModel, audioURL, audioData)
		if err != nil {
			respond("Transcription error", errors.WithStack(err))
			return
		}

		var transcriptResp struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(respBody, &transcriptResp); err != nil {
			respond("Invalid transcription response", errors.WithStack(err))
			return
		}

		respond(transcriptResp.Text, nil)
	})

	log.Printf("Listening on %s...", *port)
	log.Fatal(http.ListenAndServe(":"+*port, nil))
}

func extractURLFromText(text string) string {
	text = strings.TrimSpace(text)
	tokens := strings.Fields(text)
	for _, t := range tokens {
		if strings.HasPrefix(t, "http://") || strings.HasPrefix(t, "https://") {
			return t
		}
	}
	return ""
}

func downloadFileWithLimit(url string, maxAudioSize int64) ([]byte, error) {
	// TODO add a timeout, say, 1 minute
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("HTTP get failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.ContentLength > maxAudioSize {
		return nil, fmt.Errorf("file exceeds maximum size of %d MB", maxAudioSize/1024/1024)
	}

	limitedReader := io.LimitReader(resp.Body, maxAudioSize+1)
	buf := new(bytes.Buffer)
	n, err := buf.ReadFrom(limitedReader)
	if err != nil {
		return nil, err
	}

	if n > maxAudioSize {
		return nil, fmt.Errorf("downloaded file exceeds size limit")
	}

	return buf.Bytes(), nil
}

func sendToTranscription(whisperServerURL, whisperModel, audioURL string, audio []byte) ([]byte, int, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	audioURLFileName, err := extractFilename(audioURL)
	if err != nil {
		return nil, 0, err
	}

	part, err := writer.CreateFormFile("file", audioURLFileName)
	if err != nil {
		return nil, 0, err
	}
	part.Write(audio)

	writer.WriteField("model", whisperModel)
	writer.Close()

	req, err := http.NewRequest("POST", whisperServerURL+"/v1/audio/transcriptions", body)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, err
	}

	return respData, resp.StatusCode, nil
}

func extractFilename(input string) (string, error) {
	dotIndex := strings.LastIndex(input, ".")
	if dotIndex == -1 || dotIndex == len(input)-1 {
		return "", fmt.Errorf("invalid or missing file extension")
	}

	ext := input[dotIndex:] // includes the dot
	return "audio" + ext, nil
}
