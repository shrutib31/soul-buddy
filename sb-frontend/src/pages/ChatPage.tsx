import { useState, useRef, useEffect, useCallback } from 'react'
import type React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useBuddy } from '../contexts/BuddyContext'
import { sendMessage } from '../services/api'
import type { ChatMode, Message } from '../types'
import Header from '../components/Header'
import ChatWindow from '../components/ChatWindow'
import ChatInput from '../components/ChatInput'
import ConversationList from '../components/ConversationList'
import { PanelLeftOpen, PanelLeftClose } from 'lucide-react'

function generateId() {
  return Math.random().toString(36).slice(2)
}

export default function ChatPage() {
  const { user, token } = useAuth()
  const { config } = useBuddy()
  const navigate = useNavigate()

  const isIncognito = !user
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string | undefined>()
  const [sending, setSending] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [chatMode, setChatMode] = useState<ChatMode>('default')
  const bottomRef = useRef<HTMLDivElement>(null) as React.RefObject<HTMLDivElement>

  // Redirect to onboarding if cognito user has no config
  useEffect(() => {
    if (user && !config) {
      navigate('/onboarding')
    }
  }, [user, config, navigate])

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Greeting on mount
  useEffect(() => {
    const name = config?.name ?? 'SoulBuddy'
    const greeting = isIncognito
      ? `Hi there! I'm ${name}, your anonymous wellness companion. What's on your mind?`
      : `Welcome back! I'm ${name}. How are you feeling today?`
    setMessages([
      { id: generateId(), speaker: 'bot', text: greeting, timestamp: new Date() },
    ])
  }, [isIncognito, config?.name])

  const handleSend = useCallback(
    async (text: string) => {
      if (!text.trim() || sending) return

      const userMsg: Message = {
        id: generateId(),
        speaker: 'user',
        text: text.trim(),
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, userMsg])
      setSending(true)

      try {
        const response = await sendMessage(
          {
            message: text.trim(),
            is_incognito: isIncognito,
            sb_conv_id: conversationId,
            domain: config?.domain ?? 'general',
            chat_preference: config?.personality ?? 'general',
            chat_mode: chatMode,
          },
          token ?? undefined,
        )

        if (response.conversation_id) {
          setConversationId(response.conversation_id)
        }

        const botMsg: Message = {
          id: generateId(),
          speaker: 'bot',
          text: response.error
            ? "I'm having a little trouble right now. Please try again in a moment."
            : response.response,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, botMsg])
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            speaker: 'bot',
            text: "Sorry, I couldn't reach the server. Please check your connection and try again.",
            timestamp: new Date(),
          },
        ])
      } finally {
        setSending(false)
      }
    },
    [sending, isIncognito, conversationId, config, token, chatMode],
  )

  function startNewConversation() {
    setMessages([])
    setConversationId(undefined)
    setSidebarOpen(false)
    const name = config?.name ?? 'SoulBuddy'
    setTimeout(() => {
      setMessages([
        {
          id: generateId(),
          speaker: 'bot',
          text: `Starting fresh! I'm ${name}. What would you like to talk about?`,
          timestamp: new Date(),
        },
      ])
    }, 50)
  }

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar (cognito only) */}
      {user && (
        <ConversationList
          open={sidebarOpen}
          token={token ?? ''}
          activeConvId={conversationId}
          onSelect={(id) => {
            setConversationId(id)
            setMessages([])
            setSidebarOpen(false)
          }}
          onNew={startNewConversation}
        />
      )}

      {/* Main */}
      <div className="flex flex-col flex-1 min-w-0">
        <Header
          buddyName={config?.name ?? 'SoulBuddy'}
          isIncognito={isIncognito}
          personality={config?.personality}
          leftSlot={
            user ? (
              <button
                onClick={() => setSidebarOpen((o) => !o)}
                className="p-2 rounded-lg hover:bg-gray-100 text-gray-500"
                aria-label="Toggle sidebar"
              >
                {sidebarOpen ? (
                  <PanelLeftClose className="w-5 h-5" />
                ) : (
                  <PanelLeftOpen className="w-5 h-5" />
                )}
              </button>
            ) : null
          }
        />

        <ChatWindow messages={messages} sending={sending} bottomRef={bottomRef} />

        <ChatInput
          onSend={handleSend}
          disabled={sending}
          chatMode={chatMode}
          onModeChange={setChatMode}
        />
      </div>
    </div>
  )
}
