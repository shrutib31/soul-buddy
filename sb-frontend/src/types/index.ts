export type Domain = 'student' | 'employee' | 'corporate' | 'general'
export type ChatPreference = 'gentle_reflective' | 'direct_practical' | 'general'
export type ChatMode = 'default' | 'reflection' | 'venting' | 'therapist'

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

export interface UserMemory {
  growth_summary: string | null
  recurring_themes: string[] | null
  behavioral_patterns: string[] | null
  emotional_baseline: string | null
  preferred_modes: string[] | null
  risk_signals: Record<string, unknown> | null
  last_updated: string | null
}

export interface SessionSummary {
  session_story?: string
  emotional_arc?: string
  key_takeaways?: string[]
  dominant_emotion?: string
  recommended_next_step?: string
  risk_level?: string
}

export interface Metric {
  metric_type: string
  metric_name: string
  metric_value: number | null
  metadata: Record<string, unknown> | null
  computed_at: string | null
}

export interface WeeklyGrowth {
  weekly_growth_score: number | null
  trend: 'improving' | 'stable' | 'worsening' | null
  sessions_analysed: number | null
  computed_at: string | null
  message?: string
}

export interface ChatRequest {
  message: string
  is_incognito: boolean
  sb_conv_id?: string
  domain: Domain
  chat_preference: ChatPreference
  chat_mode: ChatMode
}

export interface ChatResponse {
  response: string
  conversation_id: string
  error?: string
}
