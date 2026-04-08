import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { BuddyProvider } from './contexts/BuddyContext'
import LoginPage from './pages/LoginPage'
import OnboardingPage from './pages/OnboardingPage'
import ChatPage from './pages/ChatPage'
import HomePage from './pages/HomePage'

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-buddy-200 border-t-buddy-600 rounded-full animate-spin" />
      </div>
    )
  }
  if (!user) return <Navigate to="/login" replace />
  return <>{children}</>
}

function AppRoutes() {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-buddy-200 border-t-buddy-600 rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/chat" replace /> : <LoginPage />} />
      <Route
        path="/onboarding"
        element={
          <RequireAuth>
            <OnboardingPage />
          </RequireAuth>
        }
      />
      <Route path="/chat" element={<ChatPage />} />
      <Route path="/home" element={<HomePage />} />
      <Route path="/" element={<Navigate to={user ? '/chat' : '/login'} replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <BuddyProvider>
          <AppRoutes />
        </BuddyProvider>
      </AuthProvider>
    </BrowserRouter>
  )
}
