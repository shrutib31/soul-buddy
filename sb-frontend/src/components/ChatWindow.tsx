import type { RefObject } from 'react'
import type { Message } from '../types'
import { useBuddy } from '../contexts/BuddyContext'
import MessageBubble, { TypingIndicator } from './MessageBubble'

interface Props {
  messages: Message[]
  sending: boolean
  bottomRef: RefObject<HTMLDivElement>
}

export default function ChatWindow({ messages, sending, bottomRef }: Props) {
  const { config } = useBuddy()
  const buddyName = config?.name ?? 'SoulBuddy'

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin px-4 py-6">
      <div className="max-w-2xl mx-auto">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} buddyName={buddyName} />
        ))}
        {sending && <TypingIndicator buddyName={buddyName} />}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
