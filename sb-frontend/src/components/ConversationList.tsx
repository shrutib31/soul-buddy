import { useEffect, useState } from 'react'
import { getConversations } from '../services/api'
import type { Conversation } from '../types'
import { MessageSquare, Plus, Loader2 } from 'lucide-react'

interface Props {
  open: boolean
  token: string
  activeConvId?: string
  onSelect: (id: string) => void
  onNew: () => void
}

export default function ConversationList({ open, token, activeConvId, onSelect, onNew }: Props) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!open || !token) return
    setLoading(true)
    getConversations(token)
      .then(setConversations)
      .finally(() => setLoading(false))
  }, [open, token])

  if (!open) return null

  return (
    <aside className="w-64 border-r border-gray-100 bg-white flex flex-col shrink-0 h-full">
      <div className="p-4 border-b border-gray-100">
        <button
          onClick={onNew}
          className="w-full py-2.5 bg-buddy-600 text-white rounded-xl text-sm font-semibold flex items-center justify-center gap-2 hover:bg-buddy-700 transition-colors"
        >
          <Plus className="w-4 h-4" /> New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-2">
        {loading ? (
          <div className="flex items-center justify-center py-8 text-gray-400">
            <Loader2 className="w-5 h-5 animate-spin" />
          </div>
        ) : conversations.length === 0 ? (
          <p className="text-xs text-gray-400 text-center py-8">No past conversations</p>
        ) : (
          conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => onSelect(conv.id)}
              className={`w-full text-left px-3 py-3 rounded-xl mb-1 transition-colors ${
                activeConvId === conv.id
                  ? 'bg-buddy-50 text-buddy-700'
                  : 'hover:bg-gray-50 text-gray-700'
              }`}
            >
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 shrink-0 text-gray-400" />
                <div className="min-w-0">
                  <p className="text-xs font-medium truncate">
                    {conv.last_message ?? 'Conversation'}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {new Date(conv.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
            </button>
          ))
        )}
      </div>
    </aside>
  )
}
