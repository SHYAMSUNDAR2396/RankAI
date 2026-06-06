import { NavLink } from 'react-router-dom';

const navItems = [
  { to: '/', icon: 'dashboard', label: 'Dashboard' },
  { to: '/candidates', icon: 'groups', label: 'Candidates' },
  { to: '/jobs', icon: 'work', label: 'Job Roles' },
  { to: '/audit', icon: 'description', label: 'Audit Report' },
  { to: '/settings', icon: 'settings', label: 'Settings' },
];

export default function SideNavBar() {
  return (
    <nav className="fixed left-0 top-0 h-screen w-[220px] bg-inverse-surface flex flex-col justify-between py-6 z-20">
      <div>
        {/* Brand */}
        <div className="px-6 mb-8 flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-primary flex items-center justify-center text-on-primary font-bold text-lg">
            R
          </div>
          <div>
            <h1 className="text-headline-sm font-bold text-primary-fixed">RankAI</h1>
            <p className="text-[10px] text-primary-fixed-dim opacity-80">AI Talent Acquisition</p>
          </div>
        </div>

        {/* Navigation */}
        <ul className="flex flex-col">
          {navItems.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `py-3 px-4 flex items-center gap-3 transition-colors text-body-md ${
                    isActive
                      ? 'bg-primary-container/20 text-primary-fixed border-l-4 border-primary-fixed font-bold opacity-90'
                      : 'text-on-surface-variant border-l-4 border-transparent hover:bg-primary-container/10 hover:text-primary-fixed'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className="material-symbols-outlined"
                      style={isActive ? { fontVariationSettings: "'FILL' 1" } : undefined}
                    >
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </div>

      {/* User */}
      <div className="px-4">
        <a
          href="#"
          className="text-on-surface-variant py-3 px-4 flex items-center gap-3 hover:text-primary-fixed transition-colors rounded"
        >
          <span className="material-symbols-outlined">account_circle</span>
          <span className="text-body-md">Aisha Patel</span>
        </a>
      </div>
    </nav>
  );
}
