import React from 'react';

export type ActiveTab = 'workbench' | 'dataset' | 'architecture' | 'code';

interface NavbarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  onExportBundle: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onSelectTab,
  onExportBundle,
}) => {
  const navItems: { id: ActiveTab; label: string }[] = [
    { id: 'workbench', label: 'Recognition' },
    { id: 'dataset', label: 'MAPPA Dataset' },
    { id: 'architecture', label: 'ResNet-50 Architecture' },
    { id: 'code', label: 'PyTorch Code' },
  ];

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between gap-3 sm:gap-6 lg:gap-8 px-3.5 sm:px-6 py-3 bg-[#0B0E17]/95 backdrop-blur border-b border-slate-800">
      {/* Zone 1: Single text element wordmark */}
      <a
        href="#workbench"
        onClick={(e) => {
          e.preventDefault();
          onSelectTab('workbench');
        }}
        className="text-sm sm:text-base font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 rounded"
      >
        JJK ResNet-50 Studio
      </a>

      {/* Zone 2: 4 clean single-line text navigation links */}
      <nav
        aria-label="Primary Workspace Navigation"
        className="flex items-center gap-3 sm:gap-6 text-xs sm:text-sm font-medium text-slate-400 overflow-x-auto no-scrollbar"
      >
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectTab(item.id)}
              aria-current={isActive ? 'page' : undefined}
              className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 ${
                isActive
                  ? 'text-cyan-400 border-cyan-400 font-semibold'
                  : 'text-slate-400 border-transparent hover:text-slate-100 hover:border-slate-700'
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Zone 3: 1 primary action */}
      <div className="flex items-center shrink-0">
        <button
          type="button"
          onClick={onExportBundle}
          className="px-2.5 sm:px-4 py-1.5 sm:py-2 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-lg hover:bg-cyan-300 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
        >
          Export PyTorch
        </button>
      </div>
    </header>
  );
};
