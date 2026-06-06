import { BrowserRouter, Routes, Route } from 'react-router-dom';
import SideNavBar from './components/SideNavBar';
import DashboardPage from './pages/DashboardPage';
import CandidateDetailPage from './pages/CandidateDetailPage';
import AuditReportPage from './pages/AuditReportPage';
import SetupPipelinePage from './pages/SetupPipelinePage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-background">
        <SideNavBar />
        <div className="flex-1 ml-[220px] flex flex-col min-w-0">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/candidates/:id" element={<CandidateDetailPage />} />
            <Route path="/candidates" element={<DashboardPage />} />
            <Route path="/audit" element={<AuditReportPage />} />
            <Route path="/pipeline" element={<SetupPipelinePage />} />
            <Route path="/jobs" element={<PlaceholderPage title="Job Roles" icon="work" />} />
            <Route path="/settings" element={<PlaceholderPage title="Settings" icon="settings" />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}

function PlaceholderPage({ title, icon }: { title: string; icon: string }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 p-8">
      <span className="material-symbols-outlined text-6xl text-outline-variant">{icon}</span>
      <h2 className="text-headline-md text-navy">{title}</h2>
      <p className="text-body-md text-on-surface-variant">Coming soon.</p>
    </div>
  );
}
