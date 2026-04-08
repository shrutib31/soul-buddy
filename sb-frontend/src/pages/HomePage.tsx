import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Heart, User, MessageCircle, LogOut,
  BookOpen, BarChart2, TrendingUp, Clock,
  ChevronDown, ChevronUp,
} from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import {
  getConversations,
  getUserMemory,
  getSessionSummary,
  getSessionMetrics,
  getWeeklyGrowth,
} from '../services/api'
import type { Conversation, UserMemory, SessionSummary, Metric, WeeklyGrowth } from '../types'
import '../styles.css'

// ── helpers ──────────────────────────────────────────────────────────────────

function formatDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

function metricLabel(name: string): string {
  const map: Record<string, string> = {
    emotional_stability_score: 'Emotional Stability',
    engagement_score: 'Engagement',
    progress_score: 'Progress',
    risk_score: 'Risk Level',
    mode_preference: 'Mode Preference',
  }
  return map[name] ?? name.replace(/_/g, ' ')
}

function trendColor(trend: string | null): string {
  if (trend === 'improving') return '#22c55e'
  if (trend === 'worsening') return '#ef4444'
  return '#f59e0b'
}

function ScoreBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <div style={{
        flex: 1, height: 8, background: '#e5e7eb', borderRadius: 99, overflow: 'hidden',
      }}>
        <div style={{ width: `${pct}%`, height: '100%', background: 'var(--primary)', borderRadius: 99 }} />
      </div>
      <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)', minWidth: 32 }}>{pct}%</span>
    </div>
  )
}

// ── section wrapper ───────────────────────────────────────────────────────────

function InsightSection({
  icon, title, children, defaultOpen = true,
}: {
  icon: React.ReactNode
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card" style={{ marginBottom: '1rem' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ color: 'var(--primary)' }}>{icon}</span>
          <span style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text-main)' }}>{title}</span>
        </div>
        {open ? <ChevronUp size={18} color="var(--text-muted)" /> : <ChevronDown size={18} color="var(--text-muted)" />}
      </button>
      {open && <div style={{ marginTop: '1rem' }}>{children}</div>}
    </div>
  )
}

// ── main component ────────────────────────────────────────────────────────────

export default function HomePage() {
  const navigate = useNavigate()
  const { token, signOut } = useAuth()

  const [conversations, setConversations] = useState<Conversation[]>([])
  const [memory, setMemory] = useState<UserMemory | null>(null)
  const [summary, setSummary] = useState<SessionSummary | null>(null)
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [weekly, setWeekly] = useState<WeeklyGrowth | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!token) { setLoading(false); return }

    async function load() {
      try {
        const [convs, mem, wk] = await Promise.all([
          getConversations(token!),
          getUserMemory(token!),
          getWeeklyGrowth(token!),
        ])
        setConversations(convs)
        setMemory(mem)
        setWeekly(wk)

        // Load last session's summary + metrics
        const lastConvId = convs[0]?.conversation_id
        if (lastConvId) {
          const [sum, mets] = await Promise.all([
            getSessionSummary(token!, lastConvId),
            getSessionMetrics(token!, lastConvId),
          ])
          setSummary(sum)
          setMetrics(mets)
        }
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [token])

  async function handleLogout() {
    await signOut()
    navigate('/login')
  }

  // deduplicate metrics: keep latest per metric_name
  const latestMetrics = Object.values(
    metrics.reduce<Record<string, Metric>>((acc, m) => {
      if (!acc[m.metric_name] || (m.computed_at ?? '') > (acc[m.metric_name].computed_at ?? '')) {
        acc[m.metric_name] = m
      }
      return acc
    }, {})
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-buddy-50 via-indigo-50 to-purple-50 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 bg-white/80 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-buddy-600 rounded-xl flex items-center justify-center shadow-lg">
            <Heart className="w-5 h-5 text-white" fill="currentColor" />
          </div>
          <span className="text-xl font-bold text-gray-900">SoulBuddy</span>
        </div>
        <button
          className="flex items-center gap-2 text-sm text-gray-500 hover:text-buddy-600 font-semibold"
          onClick={handleLogout}
        >
          <LogOut className="w-4 h-4" /> Logout
        </button>
      </header>

      <main style={{ flex: 1, maxWidth: 600, width: '100%', margin: '0 auto', padding: '1.5rem 1rem 2rem' }}>

        {/* Action buttons */}
        <div className="card" style={{ marginBottom: '1.5rem', textAlign: 'center' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.25rem' }}>Welcome back</h2>
          <p className="text-muted" style={{ marginBottom: '1.25rem' }}>Your personal mental wellness companion</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <button className="btn-primary flex items-center justify-center gap-2" onClick={() => navigate('/chat')}>
              <MessageCircle size={18} /> Start a Conversation
            </button>
            <button
              className="btn-primary flex items-center justify-center gap-2"
              style={{ background: 'var(--secondary)' }}
              onClick={() => navigate('/profile')}
            >
              <User size={18} /> View Profile
            </button>
          </div>
        </div>

        {/* Insights section */}
        <h3 style={{ fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.75rem', color: 'var(--text-main)' }}>
          Your Insights
        </h3>

        {loading ? (
          <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
            Loading your insights…
          </div>
        ) : !token ? (
          <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
            Sign in to see your personalised insights.
          </div>
        ) : (
          <>
            {/* ── Summary ── */}
            <InsightSection icon={<BookOpen size={18} />} title="Summary">
              {memory?.growth_summary ? (
                <p style={{ fontSize: '0.92rem', lineHeight: 1.6, color: 'var(--text-main)', margin: 0 }}>
                  {memory.growth_summary}
                </p>
              ) : summary?.session_story ? (
                <p style={{ fontSize: '0.92rem', lineHeight: 1.6, color: 'var(--text-main)', margin: 0 }}>
                  {summary.session_story}
                </p>
              ) : (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', margin: 0 }}>
                  Your growth summary will appear here after a few sessions.
                </p>
              )}
              {summary?.key_takeaways && summary.key_takeaways.length > 0 && (
                <ul style={{ marginTop: '0.75rem', paddingLeft: '1.25rem', fontSize: '0.88rem', color: 'var(--text-main)' }}>
                  {summary.key_takeaways.map((t, i) => <li key={i}>{t}</li>)}
                </ul>
              )}
              {summary?.recommended_next_step && (
                <div style={{
                  marginTop: '0.75rem', padding: '0.6rem 0.85rem',
                  background: '#ede9fe', borderRadius: '0.75rem',
                  fontSize: '0.88rem', color: '#5b21b6',
                }}>
                  💡 {summary.recommended_next_step}
                </div>
              )}
            </InsightSection>

            {/* ── Conversation History ── */}
            <InsightSection icon={<Clock size={18} />} title="Conversation History" defaultOpen={false}>
              {conversations.length === 0 ? (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', margin: 0 }}>
                  No conversations yet. Start chatting to build your history.
                </p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  {conversations.slice(0, 5).map(conv => (
                    <div
                      key={conv.conversation_id}
                      style={{
                        padding: '0.65rem 0.85rem',
                        background: '#f9fafb',
                        borderRadius: '0.75rem',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        cursor: 'pointer',
                        border: '1px solid #e5e7eb',
                      }}
                      onClick={() => navigate('/chat', { state: { conversationId: conv.conversation_id } })}
                    >
                      <div>
                        <div style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-main)' }}>
                          {conv.mode === 'cognito' ? 'Authenticated' : 'Incognito'} session
                        </div>
                        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 2 }}>
                          {conv.messages.length} messages · {formatDate(conv.started_at)}
                        </div>
                      </div>
                      <MessageCircle size={16} color="var(--primary)" />
                    </div>
                  ))}
                </div>
              )}
            </InsightSection>

            {/* ── Daily Insights ── */}
            <InsightSection icon={<BarChart2 size={18} />} title="Daily Insights">
              {latestMetrics.length === 0 ? (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', margin: 0 }}>
                  Daily insights appear after your first conversation.
                </p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  {latestMetrics
                    .filter(m => m.metric_name !== 'mode_preference')
                    .map(m => (
                      <div key={m.metric_name}>
                        <div style={{
                          display: 'flex', justifyContent: 'space-between',
                          fontSize: '0.85rem', marginBottom: '0.3rem',
                        }}>
                          <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{metricLabel(m.metric_name)}</span>
                          <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                            {formatDate(m.computed_at)}
                          </span>
                        </div>
                        {m.metric_value !== null
                          ? <ScoreBar value={m.metric_value} />
                          : <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>—</span>
                        }
                      </div>
                    ))}
                  {/* Mode preference */}
                  {latestMetrics.filter(m => m.metric_name === 'mode_preference').map(m => (
                    <div key="mode" style={{ marginTop: '0.25rem' }}>
                      <span style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-main)' }}>Mode Preference</span>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.4rem' }}>
                        {Object.entries((m.metadata?.mode_distribution as Record<string, number>) ?? {}).map(([mode, pct]) => (
                          <span key={mode} style={{
                            padding: '0.25rem 0.65rem', background: '#ede9fe',
                            borderRadius: 99, fontSize: '0.78rem', color: '#5b21b6', fontWeight: 600,
                          }}>
                            {mode} {Math.round(pct * 100)}%
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </InsightSection>

            {/* ── Weekly Insights ── */}
            <InsightSection icon={<TrendingUp size={18} />} title="Weekly Insights">
              {!weekly || weekly.weekly_growth_score === null ? (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', margin: 0 }}>
                  {weekly?.message ?? 'Weekly insights appear after a few sessions.'}
                </p>
              ) : (
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.75rem' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '2rem', fontWeight: 800, color: trendColor(weekly.trend) }}>
                        {weekly.trend === 'improving' ? '↑' : weekly.trend === 'worsening' ? '↓' : '→'}
                      </div>
                      <div style={{
                        fontSize: '0.8rem', fontWeight: 700, textTransform: 'capitalize',
                        color: trendColor(weekly.trend),
                      }}>
                        {weekly.trend}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Growth score</div>
                      <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-main)' }}>
                        {weekly.weekly_growth_score > 0 ? '+' : ''}{weekly.weekly_growth_score.toFixed(3)}
                      </div>
                      {weekly.sessions_analysed && (
                        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                          Based on {weekly.sessions_analysed} sessions
                        </div>
                      )}
                    </div>
                  </div>
                  {memory?.recurring_themes && memory.recurring_themes.length > 0 && (
                    <div>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                        RECURRING THEMES
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                        {memory.recurring_themes.map(t => (
                          <span key={t} style={{
                            padding: '0.25rem 0.65rem', background: '#fef3c7',
                            borderRadius: 99, fontSize: '0.78rem', color: '#92400e', fontWeight: 600,
                          }}>
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </InsightSection>
          </>
        )}
      </main>

      <footer className="text-center text-xs text-gray-400 py-4">
        © 2025 SoulBuddy. All rights reserved.
      </footer>
    </div>
  )
}
