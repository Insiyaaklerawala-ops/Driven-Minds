"use client";

import { useState, useRef, useEffect } from "react";
import { Panel } from "../ui/Panel";
import { askQuestion } from "@/lib/api";
import { Send, MessageSquare } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const STARTER_QUESTIONS = [
  "Which group is most affected?",
  "What caused this bias?",
  "Is this safe to deploy?",
];

export function ChatPanel({ sessionId }: { sessionId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(question: string) {
    if (!question.trim() || isThinking) return;
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setIsThinking(true);
    try {
      const res = await askQuestion(sessionId, question);
      setMessages((prev) => [...prev, { role: "assistant", content: res.answer }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Something went wrong answering that. Try again." },
      ]);
    } finally {
      setIsThinking(false);
    }
  }

  return (
    <Panel>
      <div className="flex items-center gap-2 mb-4">
        <MessageSquare className="w-4 h-4 text-ink-faint" />
        <span className="text-xs font-mono text-ink-muted uppercase tracking-wide">
          Ask about this report
        </span>
      </div>

      {messages.length === 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {STARTER_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => sendMessage(q)}
              className="text-xs font-mono text-ink-muted border border-border rounded-full px-3 py-1.5
                         hover:border-signal/50 hover:text-signal transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {messages.length > 0 && (
        <div className="space-y-3 mb-4 max-h-80 overflow-y-auto pr-1">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`text-sm rounded-lg px-3 py-2 leading-relaxed ${
                m.role === "user"
                  ? "bg-panel-raised text-ink-primary ml-8"
                  : "bg-signal/5 border border-signal/20 text-ink-primary mr-4"
              }`}
            >
              {m.content}
            </div>
          ))}
          {isThinking && (
            <div className="text-xs font-mono text-ink-faint mr-4 px-3">thinking...</div>
          )}
          <div ref={bottomRef} />
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage(input);
        }}
        className="flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          className="flex-1 bg-panel-raised border border-border rounded-lg px-3 py-2.5 text-sm
                     text-ink-primary placeholder:text-ink-faint
                     focus:outline-none focus:border-signal/60 focus:ring-1 focus:ring-signal/30"
        />
        <button
          type="submit"
          disabled={isThinking || !input.trim()}
          className="bg-signal text-void rounded-lg px-4 disabled:opacity-30 disabled:cursor-not-allowed
                     hover:opacity-90 transition-opacity"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </Panel>
  );
}