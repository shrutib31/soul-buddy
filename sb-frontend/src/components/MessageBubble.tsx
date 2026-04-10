import type { Message } from '../types'
import { Heart } from 'lucide-react'

interface Props {
  message: Message
  buddyName?: string
}

export default function MessageBubble({ message, buddyName = 'Buddy' }: Props) {
  const isUser = message.speaker === 'user'
  const timeStr = message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-xs lg:max-w-md">
          <div className="bg-buddy-600 text-white px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed">
            {message.text}
          </div>
          <p className="text-xs text-gray-400 text-right mt-1 pr-1">{timeStr}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-end gap-2 mb-4">
      <div className="w-7 h-7 bg-buddy-100 rounded-xl flex items-center justify-center shrink-0 mb-5">
        <Heart className="w-3.5 h-3.5 text-buddy-600" fill="currentColor" />
      </div>
      <div className="max-w-xs lg:max-w-md">
        <p className="text-xs text-gray-400 mb-1 pl-1">{buddyName}</p>
        <div className="bg-white border border-gray-100 shadow-sm px-4 py-3 rounded-2xl rounded-bl-sm text-sm leading-relaxed text-gray-700">
          {message.text}
        </div>
        <p className="text-xs text-gray-400 mt-1 pl-1">{timeStr}</p>
      </div>
    </div>
  )
}

export function TypingIndicator({ buddyName }: { buddyName?: string }) {
  return (
    <div className="flex items-end gap-2 mb-4">
      <div className="w-7 h-7 bg-buddy-100 rounded-xl flex items-center justify-center shrink-0 mb-5">
        <Heart className="w-3.5 h-3.5 text-buddy-600" fill="currentColor" />
      </div>
      <div>
        <p className="text-xs text-gray-400 mb-1 pl-1">{buddyName ?? 'Buddy'}</p>
        <div className="bg-white border border-gray-100 shadow-sm px-4 py-3 rounded-2xl rounded-bl-sm inline-flex gap-1.5 items-center">
          <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
          <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
          <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" />
        </div>
      </div>
    </div>
  )
}
