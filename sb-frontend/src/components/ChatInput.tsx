import { useState, useRef } from 'react'
import { Send } from 'lucide-react'

interface Props {
  onSend: (text: string) => void
  disabled?: boolean
}

export default function ChatInput({ onSend, disabled = false }: Props) {
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
    // Auto-resize
    const ta = e.target
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`
  }

  return (
    <div className="border-t border-gray-100 bg-white px-4 py-3 shrink-0">
      <div className="max-w-2xl mx-auto flex items-end gap-3">
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
  )
}
