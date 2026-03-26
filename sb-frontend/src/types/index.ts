export type Domain = 'student' | 'employee' | 'corporate' | 'general'
export type ChatPreference = 'gentle_reflective' | 'direct_practical' | 'general'

export interface BuddyConfig {
  name: string
  personality: ChatPreference
  domain: Domain
}

export interface Message {
  id: string
  speaker: 'user' | 'bot'
  text: string
  timestamp: Date
}

export interface ConversationMessage {
  id: string
  turn_index: number
  speaker: 'user' | 'bot'
  message: string
  created_at: string | null
}

export interface Conversation {
  conversation_id: string
  mode: string
  started_at: string | null
  ended_at: string | null
  messages: ConversationMessage[]
}

export interface ChatRequest {
  message: string
  is_incognito: boolean
  sb_conv_id?: string
  domain: Domain
  chat_preference: ChatPreference
}

export interface ChatResponse {
  response: string
  conversation_id: string
  error?: string
}
