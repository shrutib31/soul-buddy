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

export interface Conversation {
  id: string
  created_at: string
  last_message?: string
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
