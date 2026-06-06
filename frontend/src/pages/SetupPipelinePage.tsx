import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';

type Step = 1 | 2 | 3;

export default function SetupPipelinePage() {
  const [step, setStep] = useState<Step>(1);
  const [jobText, setJobText] = useState('');
  const [jobTitle, setJobTitle] = useState('');
  const [company, setCompany] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  // Detect requirements from pasted text (simple regex)
  const mustHaves = (jobText.match(/must|required|experience with|proficien/gi) || []).length;
  const niceToHaves = (jobText.match(/nice to have|preferred|bonus|ideally/gi) || []).length;
  const cultureSignals = (jobText.match(/culture|async|documentation|ownership/gi) || []).length;

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = Array.from(e.dataTransfer.files);
    setFiles((prev) => [...prev, ...dropped]);
  }, []);

  const handleRunPipeline = async () => {
    setRunning(true);
    setProgress(0);
    setError('');
    try {
      // 1. Upload Job Description
      if (jobText.trim() || jobTitle.trim() || company.trim()) {
        const jdJson = JSON.stringify({
          job_id: "custom-" + Date.now(),
          title: jobTitle || "Custom Job",
          company: company || "Custom Company",
          raw_text: jobText,
          requirements: [] // Fallback, pipeline extracts them automatically
        });
        const jdFile = new File([jdJson], "sample_job_description.json", { type: "application/json" });
        await api.uploadJobDescription(jdFile);
      }

      // 2. Upload Resumes
      if (files.length > 0) {
        await api.uploadResumes(files);
      }

      // 3. Run Pipeline
      await api.runPipeline();
      
      // Simulate progress visually
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 98) return prev;
          return prev + Math.random() * 2;
        });
      }, 1500);

      // Poll status
      const poll = setInterval(async () => {
        const status = await api.getPipelineStatus();
        if (!status.running) {
          clearInterval(poll);
          clearInterval(progressInterval);
          setRunning(false);
          setProgress(100);
          if (status.last_result?.returncode === 0) {
            navigate('/');
          } else {
            setError(status.last_result?.stderr_tail || status.last_result?.error || 'Pipeline failed');
          }
        }
      }, 3000);
    } catch (err) {
      setRunning(false);
      setError(String(err));
    }
  };

  const steps = [
    { num: 1 as Step, label: 'Job Description' },
    { num: 2 as Step, label: 'Upload Candidates' },
    { num: 3 as Step, label: 'Configure & Run' },
  ];

  return (
    <div className="flex-1 overflow-y-auto w-full h-full p-8 bg-background">
      <div className="max-w-[680px] mx-auto mt-8 mb-16 flex flex-col gap-6">
        {/* Header */}
        <div className="flex justify-between items-center mb-4 animate-fade-in" style={{ opacity: 0 }}>
          <h2 className="text-headline-md text-navy">New Pipeline</h2>
          <button
            onClick={() => navigate('/')}
            className="text-body-md text-on-surface-variant flex items-center gap-2 hover:text-navy transition-colors"
          >
            <span className="material-symbols-outlined text-sm">close</span> Cancel
          </button>
        </div>

        {/* Stepper */}
        <div className="flex items-center justify-between relative mb-8 animate-fade-in" style={{ opacity: 0, animationDelay: '0.05s' }}>
          <div className="absolute left-0 top-1/2 -translate-y-1/2 w-full h-[2px] bg-outline-variant/30 -z-10" />
          {steps.map((s) => (
            <div key={s.num} className="flex flex-col items-center gap-2 bg-background px-4">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-label-md transition-colors ${
                  step >= s.num
                    ? 'bg-primary-container text-on-primary'
                    : 'bg-surface-variant text-on-surface-variant'
                }`}
              >
                {step > s.num ? (
                  <span className="material-symbols-outlined text-[18px]">check</span>
                ) : (
                  s.num
                )}
              </div>
              <span
                className={`text-label-md ${step >= s.num ? 'text-navy' : 'text-on-surface-variant'}`}
              >
                {s.label}
              </span>
            </div>
          ))}
        </div>

        {/* Step 1: Job Description */}
        {step === 1 && (
          <div className="card p-8 flex flex-col gap-6 animate-fade-in" style={{ opacity: 0, animationDelay: '0.1s' }}>
            <h3 className="text-[18px] font-bold text-navy">Job Description</h3>

            {/* Tabs */}
            <div className="flex border-b border-outline-variant/30 gap-6">
              <button className="pb-2 border-b-2 border-primary-container text-primary-container text-label-md uppercase">
                Paste Text
              </button>
              <button className="pb-2 text-on-surface-variant text-label-md uppercase hover:text-navy transition-colors">
                Upload File
              </button>
            </div>

            <textarea
              className="w-full min-h-[200px] p-4 border border-outline-variant/50 rounded focus:border-primary-container focus:ring-1 focus:ring-primary-container text-body-md text-navy bg-surface-container-lowest resize-y outline-none"
              placeholder="Paste your full job description here..."
              value={jobText}
              onChange={(e) => setJobText(e.target.value)}
            />

            {/* Preview Strip */}
            {jobText.length > 50 && (
              <div className="flex flex-wrap gap-2 items-center bg-row-alt p-3 rounded">
                <span className="text-label-sm px-2 py-1 rounded bg-primary-container text-on-primary">
                  {mustHaves} must-haves detected
                </span>
                <span className="text-label-sm px-2 py-1 rounded bg-blue-100 text-blue-900">
                  {niceToHaves} nice-to-haves
                </span>
                <span className="text-label-sm px-2 py-1 rounded bg-purple-100 text-purple-900">
                  {cultureSignals} culture signals
                </span>
              </div>
            )}

            {/* Inputs */}
            <div className="grid grid-cols-2 gap-6">
              <div className="flex flex-col gap-2">
                <label className="text-label-md text-navy">Job Title</label>
                <input
                  className="w-full p-2 border border-outline-variant/50 rounded focus:border-primary-container focus:ring-1 focus:ring-primary-container text-body-md text-navy bg-surface-container-lowest outline-none"
                  type="text"
                  placeholder="Senior Software Engineer"
                  value={jobTitle}
                  onChange={(e) => setJobTitle(e.target.value)}
                />
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-label-md text-navy">Company Name</label>
                <input
                  className="w-full p-2 border border-outline-variant/50 rounded focus:border-primary-container focus:ring-1 focus:ring-primary-container text-body-md text-navy bg-surface-container-lowest outline-none"
                  type="text"
                  placeholder="Acme Inc"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                />
              </div>
            </div>

            {/* Footer Actions */}
            <div className="flex justify-end mt-4 pt-4 border-t border-outline-variant/30">
              <button
                onClick={() => setStep(2)}
                disabled={!jobText.trim()}
                className={`px-6 py-2 rounded text-label-md uppercase tracking-wider flex items-center gap-2 transition-opacity ${
                  !jobText.trim()
                    ? 'bg-surface-variant text-on-surface-variant opacity-50 cursor-not-allowed'
                    : 'bg-primary-container text-on-primary hover:opacity-90'
                }`}
              >
                Next: Upload Candidates{' '}
                <span className="material-symbols-outlined text-sm">arrow_forward</span>
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Upload Candidates */}
        {step === 2 && (
          <div className="card p-8 flex flex-col gap-6 animate-fade-in" style={{ opacity: 0, animationDelay: '0.1s' }}>
            <h3 className="text-[18px] font-bold text-navy">Upload Candidate Resumes</h3>

            {/* Drop zone */}
            <div
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                dragging
                  ? 'border-primary bg-primary/5'
                  : 'border-outline-variant hover:border-primary/50'
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <span className="material-symbols-outlined text-4xl text-outline mb-2 block">
                cloud_upload
              </span>
              <p className="text-body-lg text-on-surface-variant mb-1">
                Drag & drop files here, or click to browse
              </p>
              <p className="text-label-sm text-outline">PDF, DOCX, or JSON — up to 50 files</p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.json"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) setFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
                }}
              />
            </div>

            {/* File list */}
            {files.length > 0 && (
              <div className="flex flex-col gap-2">
                <span className="text-label-md text-navy">{files.length} file(s) selected</span>
                <div className="max-h-32 overflow-y-auto space-y-1">
                  {files.map((f, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between bg-row-alt px-3 py-2 rounded text-body-md"
                    >
                      <span className="flex items-center gap-2">
                        <span className="material-symbols-outlined text-[16px] text-outline">
                          description
                        </span>
                        {f.name}
                      </span>
                      <button
                        onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                        className="text-outline hover:text-error transition-colors"
                      >
                        <span className="material-symbols-outlined text-[16px]">close</span>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Footer */}
            <div className="flex justify-between mt-4 pt-4 border-t border-outline-variant/30">
              <button
                onClick={() => setStep(1)}
                className="text-on-surface-variant text-label-md hover:text-navy transition-colors flex items-center gap-1"
              >
                <span className="material-symbols-outlined text-sm">arrow_back</span> Back
              </button>
              <button
                onClick={() => setStep(3)}
                disabled={files.length === 0}
                className={`px-6 py-2 rounded text-label-md uppercase tracking-wider flex items-center gap-2 transition-opacity ${
                  files.length === 0
                    ? 'bg-surface-variant text-on-surface-variant opacity-50 cursor-not-allowed'
                    : 'bg-primary-container text-on-primary hover:opacity-90'
                }`}
              >
                Next: Configure & Run{' '}
                <span className="material-symbols-outlined text-sm">arrow_forward</span>
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Configure & Run */}
        {step === 3 && (
          running ? (
            <div className="card p-10 flex flex-col items-center text-center max-w-[560px] mx-auto animate-fade-in">
              <div className="relative flex items-center justify-center mb-8">
                <svg className="w-32 h-32 transform -rotate-90">
                  <circle className="text-surface-container-high" cx="64" cy="64" fill="transparent" r="58" stroke="currentColor" strokeWidth="8"></circle>
                  <circle 
                    className="text-primary-container transition-all duration-300 ease-out" 
                    cx="64" cy="64" fill="transparent" r="58" stroke="currentColor" 
                    strokeDasharray="364.4" 
                    strokeDashoffset={364.4 - (progress / 100) * 364.4} 
                    strokeWidth="8"
                  ></circle>
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="font-display-lg text-display-lg text-on-surface leading-none">
                    {Math.round(progress)}%
                  </span>
                </div>
              </div>
              <div className="mb-10">
                <h1 className="font-headline-md text-headline-md text-inverse-surface mb-2">Analyzing Candidates...</h1>
                <p className="font-body-md text-body-md text-on-surface-variant max-w-[400px]">
                  RankAI is evaluating <span className="font-bold text-on-surface">{files.length || 15} candidates</span> against your '{jobTitle || 'Job Role'}' job description. This usually takes a few minutes.
                </p>
              </div>
              <div className="w-full text-left space-y-4 mb-10">
                <div className="flex items-center gap-4 py-3 px-4 bg-surface-container-low rounded">
                  <span className="material-symbols-outlined text-primary-container" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                  <div className="flex-1 flex justify-between items-center">
                    <span className="font-body-md text-body-md text-on-surface">Parsing candidate resumes...</span>
                    <span className="font-label-md text-label-md text-primary-container uppercase">Complete</span>
                  </div>
                </div>
                <div className="flex items-center gap-4 py-3 px-4 border-l-4 border-primary-container bg-surface-container-high/50">
                  <span className="material-symbols-outlined text-primary-container animate-spin">sync</span>
                  <div className="flex-1 flex justify-between items-center">
                    <span className="font-body-md text-body-md text-on-surface font-bold">Running multi-perspective AI scoring...</span>
                    <span className="font-label-md text-label-md text-primary-container uppercase">In Progress</span>
                  </div>
                </div>
                <div className="flex items-center gap-4 py-3 px-4 opacity-50">
                  <span className="material-symbols-outlined text-outline">circle</span>
                  <div className="flex-1 flex justify-between items-center">
                    <span className="font-body-md text-body-md text-on-surface-variant">Performing fairness audit & bias checks...</span>
                    <span className="font-label-md text-label-md text-outline uppercase">Pending</span>
                  </div>
                </div>
                <div className="flex items-center gap-4 py-3 px-4 opacity-50">
                  <span className="material-symbols-outlined text-outline">circle</span>
                  <div className="flex-1 flex justify-between items-center">
                    <span className="font-body-md text-body-md text-on-surface-variant">Generating trajectory signals...</span>
                    <span className="font-label-md text-label-md text-outline uppercase">Pending</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="card p-8 flex flex-col gap-6 animate-fade-in" style={{ opacity: 0, animationDelay: '0.1s' }}>
              <h3 className="text-[18px] font-bold text-navy">Configure & Run Pipeline</h3>

              <div className="bg-row-alt p-4 rounded space-y-3">
                <div className="flex justify-between text-body-md">
                  <span className="text-on-surface-variant">Job Title</span>
                  <span className="text-navy font-medium">{jobTitle || 'Not specified'}</span>
                </div>
                <div className="flex justify-between text-body-md">
                  <span className="text-on-surface-variant">Company</span>
                  <span className="text-navy font-medium">{company || 'Not specified'}</span>
                </div>
                <div className="flex justify-between text-body-md">
                  <span className="text-on-surface-variant">Candidates</span>
                  <span className="text-navy font-medium">{files.length} file(s)</span>
                </div>
                <div className="flex justify-between text-body-md">
                  <span className="text-on-surface-variant">Fairness Audit</span>
                  <span className="text-primary font-medium">Enabled</span>
                </div>
              </div>

              {error && (
                <div className="bg-error-container text-on-error-container p-3 rounded text-body-md">
                  {error}
                </div>
              )}

              {/* Footer */}
              <div className="flex justify-between mt-4 pt-4 border-t border-outline-variant/30">
                <button
                  onClick={() => setStep(2)}
                  className="text-on-surface-variant text-label-md hover:text-navy transition-colors flex items-center gap-1"
                  disabled={running}
                >
                  <span className="material-symbols-outlined text-sm">arrow_back</span> Back
                </button>
                <button
                  onClick={handleRunPipeline}
                  disabled={running}
                  className="bg-primary-container text-on-primary px-6 py-2 rounded text-label-md uppercase tracking-wider hover:opacity-90 transition-opacity flex items-center gap-2 disabled:opacity-50"
                >
                  Start Pipeline
                  <span className="material-symbols-outlined text-sm">play_arrow</span>
                </button>
              </div>
            </div>
          )
        )}

        {/* Helper Text */}
        <p className="text-center text-[12px] text-on-surface-variant flex items-center justify-center gap-1">
          <span className="material-symbols-outlined text-[14px]">lock</span>
          Your data stays local. Nothing is sent to external servers.
        </p>
      </div>
    </div>
  );
}
