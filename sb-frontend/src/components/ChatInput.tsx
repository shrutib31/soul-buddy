import { useState, useRef } from 'react'
import { Send } from 'lucide-react'
import type { ChatMode } from '../types'

const MODES: { value: ChatMode; label: string; emoji: string; description: string }[] = [
  { value: 'default',    label: 'Friendly',   emoji: '😊', description: 'Warm, supportive companion' },
  { value: 'reflection', label: 'Reflect',    emoji: '🪞', description: 'Guided self-exploration' },
  { value: 'venting',    label: 'Vent',       emoji: '💬', description: 'Just here to listen' },
  { value: 'therapist',  label: 'Therapist',  emoji: '🧠', description: 'Structured CBT-style support' },
]

interface Props {
  onSend: (text: string) => void
  disabled?: boolean
  chatMode: ChatMode
  onModeChange: (mode: ChatMode) => void
}

export default function ChatInput({ onSend, disabled = false, chatMode, onModeChange }: Props) {
  const [text, setText] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function handleSend() {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setText(e.target.value)
    const ta = e.target
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`
  }

  const activeMode = MODES.find((m) => m.value === chatMode)!

  return (
    <div className="border-t border-gray-100 bg-white px-4 pt-3 pb-3 shrink-0">
      <div className="max-w-2xl mx-auto">
        {/* Mode selector */}
        <div className="flex items-center gap-1.5 mb-2.5">
          {MODES.map((m) => {
            const active = chatMode === m.value
            return (
              <button
                key={m.value}
                type="button"
                onClick={() => onModeChange(m.value)}
                title={m.description}
                className={`group flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 border ${
                  active
                    ? 'bg-buddy-600 text-white border-buddy-600 shadow-sm shadow-buddy-200'
                    : 'bg-gray-50 text-gray-500 border-gray-200 hover:border-buddy-300 hover:text-buddy-600 hover:bg-buddy-50'
                }`}
              >
                <span className="text-sm leading-none">{m.emoji}</span>
                {m.label}
              </button>
            )
          })}
          <span className="ml-auto text-[11px] text-gray-400 hidden sm:block">
            {activeMode.description}
          </span>
        </div>

        {/* Input row */}
        <div className="flex items-end gap-3">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Share what's on your mind… (Enter to send)"
            rows={1}
            disabled={disabled}
            className="flex-1 resize-none rounded-xl border border-gray-200 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-buddy-400 disabled:opacity-60 bg-gray-50 max-h-[120px] leading-relaxed"
          />
          <button
            onClick={handleSend}
            disabled={!text.trim() || disabled}
            className="w-10 h-10 bg-buddy-600 text-white rounded-xl flex items-center justify-center hover:bg-buddy-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0"
            aria-label="Send message"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        <p className="text-xs text-gray-400 text-center mt-2">
          SoulBuddy is a supportive companion, not a substitute for professional mental health care.
        </p>
      </div>
    </div>
  )
}
