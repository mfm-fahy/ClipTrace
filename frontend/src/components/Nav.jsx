import { NavLink } from 'react-router-dom'
import { Film, Upload, Search, ShieldCheck, DollarSign } from 'lucide-react'

const links = [
  { to: '/',          label: 'Dashboard',    Icon: Film },
  { to: '/register',  label: 'Register',     Icon: Upload },
  { to: '/match',     label: 'Match Clip',   Icon: Search },
  { to: '/verify',    label: 'Verify',       Icon: ShieldCheck },
  { to: '/monetize',  label: 'Monetize',     Icon: DollarSign },
]

export default function Nav() {
  return (
    <nav className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 flex items-center gap-2 h-14">
        {/* Logo */}
        <span className="flex items-center gap-2 font-bold text-brand-500 text-lg mr-6 select-none">
          <Film size={22} />
          ClipTrace
        </span>

        {links.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
               ${isActive
                 ? 'bg-brand-600 text-white'
                 : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'}`
            }
          >
            <Icon size={15} />
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
