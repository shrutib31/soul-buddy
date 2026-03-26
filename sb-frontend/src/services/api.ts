import type { ChatRequest, ChatResponse, Conversation } from '../types'

const API_BASE = '/api/v1'

async function getAuthHeaders(token?: string): Promise<HeadersInit> {
  const headers: HeadersInit = { 'Content-Type': 'application/json' }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  return headers
}

export async function sendMessage(
  req: ChatRequest,
  token?: string,
): Promise<ChatResponse> {
  const headers = await getAuthHeaders(token)
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Chat request failed (${res.status}): ${text}`)
  }
  return res.json() as Promise<ChatResponse>
}

export async function* streamMessage(
  req: ChatRequest,
  token?: string,
): AsyncGenerator<string> {
  const headers = await getAuthHeaders(token)
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers,
    body: JSON.stringify(req),
  })
  if (!res.ok || !res.body) {
    throw new Error(`Stream request failed (${res.status})`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim()
        if (data && data !== '[DONE]') yield data
      }
    }
  }
}

export async function getConversations(token: string): Promise<Conversation[]> {
  const headers = await getAuthHeaders(token)
  const res = await fetch(`${API_BASE}/chat/conversations/messages`, { headers })
  if (!res.ok) return []
  const data = await res.json() as { conversations?: Conversation[] }
  return data.conversations ?? []
}
