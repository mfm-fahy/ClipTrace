import { Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Dashboard from './pages/Dashboard'
import Register from './pages/Register'
import Match from './pages/Match'
import Verify from './pages/Verify'
import Monetization from './pages/Monetization'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Nav />
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8">
        <Routes>
          <Route path="/"            element={<Dashboard />} />
          <Route path="/register"    element={<Register />} />
          <Route path="/match"       element={<Match />} />
          <Route path="/verify"      element={<Verify />} />
          <Route path="/monetize"    element={<Monetization />} />
        </Routes>
      </main>
    </div>
  )
}
