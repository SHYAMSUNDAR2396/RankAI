import { useNavigate } from 'react-router-dom';

export default function EmptyStatePage() {
  const navigate = useNavigate();
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6 text-center px-8">
      <div className="w-16 h-16 rounded-2xl bg-teal-50 flex items-center justify-center">
        {/* Simple document stack icon in teal */}
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
          <rect x="6" y="10" width="20" height="16" rx="3" fill="#0F6E56" opacity="0.15"/>
          <rect x="4" y="7" width="20" height="16" rx="3" fill="#0F6E56" opacity="0.25"/>
          <rect x="6" y="4" width="20" height="16" rx="3" fill="#0F6E56" opacity="0.9"/>
          <circle cx="24" cy="6" r="4" fill="#EF9F27"/>
          <path d="M22.5 6l1 1 2-2" stroke="white" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">
          No candidates ranked yet
        </h1>
        <p className="text-gray-500 max-w-md leading-relaxed">
          Add a job description and upload your candidate files to run your 
          first AI-powered ranking pipeline. Results appear here in minutes.
        </p>
      </div>
      <div className="flex flex-col gap-3 w-full max-w-xs">
        <button
          onClick={() => navigate('/pipeline')}
          className="w-full py-3 px-6 bg-teal-700 text-white rounded-xl font-medium hover:bg-teal-800 transition-colors cursor-pointer"
        >
          Run your first pipeline →
        </button>
        <button
          onClick={() => navigate('/pipeline?demo=true')}
          className="w-full py-3 px-6 border border-teal-700 text-teal-700 rounded-xl font-medium hover:bg-teal-50 transition-colors cursor-pointer"
        >
          Load sample data
        </button>
      </div>
      <div className="flex gap-8 text-sm text-gray-400 mt-2">
        <span>Privacy-first — runs locally</span>
        <span>Powered by Llama 3.2</span>
        <span>Fairness audit included</span>
      </div>
    </div>
  );
}
