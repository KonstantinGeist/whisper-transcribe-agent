package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
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

type TranscriptionPageData struct {
	Text  string
	Error string
}

func main() {
	fmt.Println("whisper-transcribe-agent - supports Chat API and direct uploads")

	apiPort := flag.String("port", "8080", "API HTTP server listen port")
	uiPort := flag.String("ui-port", "7500", "UI HTTP server listen port")
	baseURL := flag.String("whisper-server-url", "", "Base URL of transcription service")
	whisperModel := flag.String("whisper-model", "", "Whisper model to use")
	maxAudioSize := flag.Int64("max-audio-size", 0, "Maximum audio file size in bytes")
	flag.Parse()

	if *baseURL == "" || *whisperModel == "" || *maxAudioSize == 0 {
		log.Fatal("All flags --whisper-server-url, --whisper-model, and --max-audio-size must be set")
	}

	go func() {
		http.HandleFunc("/", serveUploadForm)
		http.HandleFunc("/transcribe/upload", func(w http.ResponseWriter, r *http.Request) {
			uploadHandler(w, r, *baseURL, *whisperModel, *maxAudioSize)
		})
		log.Printf("UI server listening on :%s...", *uiPort)
		http.ListenAndServe(":"+*uiPort, nil)
	}()

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

	log.Printf("API server listening on :%s...", *apiPort)
	log.Fatal(http.ListenAndServe(":"+*apiPort, nil))
}

func serveUploadForm(w http.ResponseWriter, r *http.Request) {
	html := `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Whisper Transcription</title>
  <style>
    body { font-family: sans-serif; padding: 2rem; background: #f0f2f5; }
    h2 { color: #333; }
    form { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    input[type=file], input[type=submit] { display: block; margin: 1rem 0; padding: 0.5rem; }
    #processing { color: #007bff; margin-top: 1rem; display: none; }
  </style>
  <script>
    function showProcessing() {
      document.getElementById("processing").style.display = "block";
    }
  </script>
</head>
<body>
  <h2>Upload Audio File for Transcription</h2>
  <form action="/transcribe/upload" method="post" enctype="multipart/form-data" onsubmit="showProcessing()">
    <input type="file" name="file" accept="audio/*" required>
    <input type="submit" value="Upload">
  </form>
  <div id="processing">Processing...</div>
</body>
</html>`
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(html))
}

func uploadHandler(w http.ResponseWriter, r *http.Request, whisperURL, model string, maxSize int64) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST supported", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxSize)
	err := r.ParseMultipartForm(maxSize)
	if err != nil {
		http.Error(w, "File too large", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Missing file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	buf := new(bytes.Buffer)
	_, err = io.Copy(buf, file)
	if err != nil {
		http.Error(w, "Failed to read file", http.StatusInternalServerError)
		return
	}

	data, _, err := sendToTranscription(whisperURL, model, header.Filename, buf.Bytes())
	if err != nil {
		tmpl := template.Must(template.New("result").Parse(`<html><body><h3>Error: {{.Error}}</h3></body></html>`))
		tmpl.Execute(w, TranscriptionPageData{Error: err.Error()})
		return
	}

	var result struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		tmpl := template.Must(template.New("result").Parse(`<html><body><h3>Failed to parse response: {{.Error}}</h3></body></html>`))
		tmpl.Execute(w, TranscriptionPageData{Error: err.Error()})
		return
	}

	tmpl := template.Must(template.New("result").Parse(`
<html>
  <head>
    <meta charset="UTF-8">
    <title>Transcription Result</title>
    <style>
      body { font-family: sans-serif; padding: 2rem; background: #f0f2f5; }
      .container { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
      .buttons { margin-top: 1rem; }
      button { padding: 0.5rem 1rem; font-size: 1rem; }
      .text-block { white-space: pre-wrap; word-wrap: break-word; background: #f7f7f7; padding: 1rem; border-radius: 5px; }
    </style>
    <script>
      function copyText() {
        const text = document.getElementById("transcription-raw").value;
        navigator.clipboard.writeText(text).then(() => {
          alert("Copied to clipboard!");
        }, () => {
          alert("Failed to copy text.");
        });
      }
    </script>
  </head>
  <body>
    <div class="container">
      <h2>Transcription Result</h2>
      <div class="text-block" id="transcription-html">{{.Text}}</div>
      <textarea id="transcription-raw" style="display:none">{{.Text}}</textarea>
      <div class="buttons">
        <button onclick="copyText()">Copy</button>
        <button onclick="history.back()">Back</button>
      </div>
    </div>
  </body>
</html>`))

	tmpl.Execute(w, TranscriptionPageData{Text: result.Text})
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
