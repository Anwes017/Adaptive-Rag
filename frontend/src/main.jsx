import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { FileUp, Loader2, MessageSquarePlus, Send, Trash2, UserRound, Bot, ChevronDown } from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const USER_ID = "anwes";

function generateThreadId() {
  return crypto.randomUUID();
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};

  if (!response.ok) {
    const message = data.detail || data.error || response.statusText;
    throw new Error(message);
  }

  return data;
}

async function readEventStream(response, onEvent) {
  if (!response.body) {
    throw new Error("Streaming is not supported by this browser");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const rawEvent of events) {
      const lines = rawEvent.split("\n");
      const eventName = lines.find((line) => line.startsWith("event:"))?.slice(6).trim() || "message";
      const dataLine = lines.find((line) => line.startsWith("data:"));
      const data = dataLine ? JSON.parse(dataLine.slice(5).trim()) : {};
      onEvent(eventName, data);
    }
  }

  if (buffer.trim()) {
    const lines = buffer.split("\n");
    const eventName = lines.find((line) => line.startsWith("event:"))?.slice(6).trim() || "message";
    const dataLine = lines.find((line) => line.startsWith("data:"));
    const data = dataLine ? JSON.parse(dataLine.slice(5).trim()) : {};
    onEvent(eventName, data);
  }
}

function Sidebar({
  threads,
  activeThreadId,
  mode,
  uploadStatus,
  isUploading,
  onUpload,
  onNewChat,
  onSelectThread,
  onDeleteThread,
  onModeChange,
}) {
  const fileInputRef = useRef(null);

  return (
    <aside className="sidebar">
      <div className="sidebarHeader">
        <div>
          <h1>Adaptive RAG</h1>
        </div>
        <button className="iconButton" onClick={onNewChat} title="New chat" aria-label="New chat">
          <MessageSquarePlus size={20} />
        </button>
      </div>

      <input
        ref={fileInputRef}
        className="hiddenInput"
        type="file"
        accept="application/pdf,.pdf"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onUpload(file);
          event.target.value = "";
        }}
      />

      <div className="modeSwitch" role="tablist" aria-label="Chat mode">
        <button
          className={mode === "chat" ? "modeButton active" : "modeButton"}
          onClick={() => onModeChange("chat")}
          aria-pressed={mode === "chat"}
          type="button"
        >
          Normal chat
        </button>
        <button
          className={mode === "crag" ? "modeButton active" : "modeButton"}
          onClick={() => onModeChange("crag")}
          aria-pressed={mode === "crag"}
          type="button"
        >
          PDF CRAG
        </button>
      </div>

      <button className="uploadButton" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
        {isUploading ? <Loader2 className="spin" size={18} /> : <FileUp size={18} />}
        <span>{isUploading ? "Indexing PDF" : "Upload PDF"}</span>
      </button>

      {uploadStatus && <p className={uploadStatus.type === "error" ? "status error" : "status"}>{uploadStatus.text}</p>}

      <div className="threadSection">
        <h2>My Conversations</h2>
        <div className="threadList">
          {threads.length === 0 && <p className="emptyText">No saved conversations yet.</p>}
          {[...threads].reverse().map((threadId, index) => (
            <div key={threadId} className={threadId === activeThreadId ? "threadItem active" : "threadItem"}>
              <button
                className="threadButton"
                onClick={() => onSelectThread(threadId)}
                title={threadId}
                type="button"
              >
                Conversation {index + 1}
              </button>
              <button
                className="threadDeleteButton"
                onClick={(event) => {
                  event.stopPropagation();
                  onDeleteThread(threadId);
                }}
                title="Delete conversation"
                aria-label={`Delete Conversation ${index + 1}`}
                type="button"
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

function Message({ message, showDebug }) {
  const isAssistant = message.role === "assistant";
  const debug = message.debug;
  const status = message.status;

  return (
    <article className={isAssistant ? "message assistant" : "message user"}>
      <div className="avatar" aria-hidden="true">
        {isAssistant ? <Bot size={18} /> : <UserRound size={18} />}
      </div>
      <div className="messageBody">
        <div className="messageContent">
          {message.content}
          {message.streaming && <span className="streamCursor" aria-hidden="true" />}
          {message.streaming && !message.content && (
            <span className="typing">
              <Loader2 className="spin" size={18} />
              {status ? `Working on ${status}` : "Thinking"}
            </span>
          )}
        </div>
        {message.streaming && status && message.content && (
          <div className="messageStatus">
            <Loader2 className="spin" size={14} />
            Working on {status}
          </div>
        )}
        {showDebug && debug && (
          <details className="debugPanel">
            <summary>
              Debug Details
              <ChevronDown size={16} />
            </summary>
            <dl>
              <div>
                <dt>Route</dt>
                <dd>{debug.route || "-"}</dd>
              </div>
              <div>
                <dt>Verdict</dt>
                <dd>{debug.verdict || "-"}</dd>
              </div>
              <div>
                <dt>Reason</dt>
                <dd>{debug.reason || "-"}</dd>
              </div>
              <div>
                <dt>Web Query</dt>
                <dd>{debug.web_query || "-"}</dd>
              </div>
            </dl>
          </details>
        )}
      </div>
    </article>
  );
}

function App() {
  const [threadId, setThreadId] = useState(generateThreadId);
  const [threads, setThreads] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState("chat");
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const messagesEndRef = useRef(null);

  const canSend = useMemo(() => input.trim().length > 0 && !isSending, [input, isSending]);

  async function refreshThreads() {
    try {
      const data = await requestJson(`${API_URL}/threads`);
      setThreads(data.threads || []);
    } catch {
      setThreads([]);
    }
  }

  async function loadThread(threadIdToLoad) {
    setThreadId(threadIdToLoad);
    setMessages([]);

    try {
      const data = await requestJson(`${API_URL}/threads/${threadIdToLoad}`);
      setMode(data.mode || "chat");
      setMessages((data.messages || []).map((message) => ({ ...message })));
    } catch {
      setMessages([]);
    }
  }

  useEffect(() => {
    refreshThreads();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  function handleNewChat() {
    setThreadId(generateThreadId());
    setMessages([]);
  }

  function handleSelectThread(nextThreadId) {
    void loadThread(nextThreadId);
  }

  async function handleDeleteThread(threadIdToDelete) {
    try {
      await requestJson(`${API_URL}/threads/${threadIdToDelete}`, {
        method: "DELETE",
      });

      setThreads((current) => current.filter((threadId) => threadId !== threadIdToDelete));

      if (threadIdToDelete === threadId) {
        setThreadId(generateThreadId());
        setMessages([]);
      }
    } catch (error) {
      setUploadStatus({ type: "error", text: `Delete error: ${error.message}` });
    }
  }

  function handleModeChange(nextMode) {
    setMode(nextMode);
  }

  async function handleUpload(file) {
    setIsUploading(true);
    setUploadStatus(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const data = await requestJson(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      setUploadStatus({ type: "success", text: data.message || `${file.name} uploaded and indexed.` });
      setMode("crag");
    } catch (error) {
      setUploadStatus({ type: "error", text: `Upload error: ${error.message}` });
    } finally {
      setIsUploading(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const question = input.trim();
    if (!question || isSending) return;

    setInput("");
    setIsSending(true);
    const assistantMessageId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      { role: "user", content: question },
      { id: assistantMessageId, role: "assistant", content: "", streaming: true },
    ]);

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: question,
          thread_id: threadId,
          user_id: USER_ID,
          mode,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }

      await readEventStream(response, (eventName, data) => {
        if (eventName === "token") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantMessageId
                ? { ...message, content: `${message.content}${data.token || ""}` }
                : message
            )
          );
        }

        if (eventName === "status") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantMessageId
                ? { ...message, status: data.status || "thinking" }
                : message
            )
          );
        }

        if (eventName === "done") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantMessageId
                ? {
                    ...message,
                    content: message.content || data.answer || "",
                    streaming: false,
                    status: null,
                    debug:
                      mode === "crag"
                        ? {
                            route: data.route,
                            verdict: data.verdict,
                            reason: data.reason,
                            web_query: data.web_query,
                          }
                        : null,
                  }
                : message
            )
          );
        }

        if (eventName === "error") {
          throw new Error(data.error || "Streaming failed");
        }
      });

      refreshThreads();
    } catch (error) {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessageId
            ? { ...message, content: `API error: ${error.message}`, streaming: false }
            : message
        )
      );
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="appShell">
      <Sidebar
        threads={threads}
        activeThreadId={threadId}
        uploadStatus={uploadStatus}
        isUploading={isUploading}
        onUpload={handleUpload}
        onNewChat={handleNewChat}
        onSelectThread={handleSelectThread}
        onDeleteThread={handleDeleteThread}
        mode={mode}
        onModeChange={handleModeChange}
      />

      <main className="chatShell">
        <header className="chatHeader">
          <div>
            <h2>Adaptive RAG</h2>
          </div>
          <div className="headerMeta">
            <div className="modePill">{mode === "crag" ? "PDF CRAG" : "Normal chat"}</div>
            <div className="threadBadge" title={threadId}>Conversation</div>
          </div>
        </header>

        <section className="messages" aria-live="polite">
          {messages.length === 0 && (
            <div className="emptyState">
              <h3>{mode === "crag" ? "Ask about your PDF..." : "Say hello..."}</h3>
              <p>
                {mode === "crag"
                  ? "Upload PDFs from the sidebar, then ask questions about the document."
                  : "Use normal chat for quick questions without document search."}
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <Message key={`${message.role}-${index}`} message={message} showDebug={mode === "crag"} />
          ))}

          <div ref={messagesEndRef} />
        </section>

        <form className="composer" onSubmit={handleSubmit}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={mode === "crag" ? "Ask about the PDF..." : "Ask anything..."}
            disabled={isSending}
          />
          <button className="sendButton" disabled={!canSend} title="Send" aria-label="Send">
            {isSending ? <Loader2 className="spin" size={20} /> : <Send size={20} />}
          </button>
        </form>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
