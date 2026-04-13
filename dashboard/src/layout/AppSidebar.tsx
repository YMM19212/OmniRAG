import { useMemo } from "react";
import { Link, useLocation } from "react-router";
import {
  BoltIcon,
  BoxCubeIcon,
  FolderIcon,
  GridIcon,
  HorizontaLDots,
  PlugInIcon,
  TableIcon,
  VideoIcon,
} from "../icons";
import { useSidebar } from "../context/SidebarContext";
import { useI18n } from "../context/I18nContext";

const AppSidebar: React.FC = () => {
  const { isExpanded, isMobileOpen, isHovered, setIsHovered } = useSidebar();
  const location = useLocation();
  const { locale } = useI18n();
  const sections = useMemo(
    () => [
      {
        title: locale === "zh" ? "工作台" : "Workspace",
        items: [
          { name: locale === "zh" ? "总览" : "Overview", path: "/", icon: <GridIcon /> },
          { name: locale === "zh" ? "连接设置" : "Connection", path: "/settings", icon: <PlugInIcon /> },
          { name: locale === "zh" ? "导入" : "Ingest", path: "/ingest", icon: <FolderIcon /> },
        ],
      },
      {
        title: locale === "zh" ? "检索" : "Retrieval",
        items: [
          { name: locale === "zh" ? "基础检索" : "Search", path: "/search", icon: <BoltIcon /> },
          { name: locale === "zh" ? "混合检索" : "Hybrid Search", path: "/hybrid-search", icon: <VideoIcon /> },
          { name: locale === "zh" ? "记录管理" : "Records", path: "/records", icon: <TableIcon /> },
        ],
      },
      {
        title: locale === "zh" ? "平台" : "Platform",
        items: [{ name: locale === "zh" ? "能力概览" : "Capabilities", path: "/", icon: <BoxCubeIcon /> }],
      },
    ],
    [locale],
  );

  const isActive = (path: string) => location.pathname === path;

  return (
    <aside
      className={`fixed mt-16 flex flex-col lg:mt-0 top-0 px-5 left-0 bg-white dark:bg-gray-900 dark:border-gray-800 text-gray-900 h-screen transition-all duration-300 ease-in-out z-50 border-r border-gray-200 
        ${
          isExpanded || isMobileOpen
            ? "w-[290px]"
            : isHovered
            ? "w-[290px]"
            : "w-[90px]"
        }
        ${isMobileOpen ? "translate-x-0" : "-translate-x-full"}
        lg:translate-x-0`}
      onMouseEnter={() => !isExpanded && setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        className={`py-8 flex ${
          !isExpanded && !isHovered ? "lg:justify-center" : "justify-start"
        }`}
      >
        <Link to="/" className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-brand-500 text-lg font-semibold text-white shadow-theme-sm">
            O
          </div>
          {(isExpanded || isHovered || isMobileOpen) && (
            <div>
              <p className="text-base font-semibold text-gray-900 dark:text-white">OmniRAG</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Multimodal knowledge base</p>
            </div>
          )}
        </Link>
      </div>
      <div className="flex flex-col overflow-y-auto duration-300 ease-linear no-scrollbar">
        <nav className="mb-6">
          <div className="flex flex-col gap-6">
            {sections.map((section) => (
              <div key={section.title}>
              <h2
                className={`mb-4 text-xs uppercase flex leading-[20px] text-gray-400 ${
                  !isExpanded && !isHovered
                    ? "lg:justify-center"
                    : "justify-start"
                }`}
              >
                {isExpanded || isHovered || isMobileOpen ? (
                  section.title
                ) : (
                  <HorizontaLDots className="size-6" />
                )}
              </h2>
                <ul className="flex flex-col gap-2">
                  {section.items.map((item) => (
                    <li key={item.name}>
                      <Link
                        to={item.path}
                        className={`menu-item group ${
                          isActive(item.path) ? "menu-item-active" : "menu-item-inactive"
                        } ${!isExpanded && !isHovered ? "lg:justify-center" : ""}`}
                      >
                        <span
                          className={`menu-item-icon-size ${
                            isActive(item.path)
                              ? "menu-item-icon-active"
                              : "menu-item-icon-inactive"
                          }`}
                        >
                          {item.icon}
                        </span>
                        {(isExpanded || isHovered || isMobileOpen) && (
                          <span className="menu-item-text">{item.name}</span>
                        )}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </nav>
        {isExpanded || isHovered || isMobileOpen ? (
          <div className="space-y-4">
            <div className="rounded-2xl bg-brand-500 px-4 py-4 text-white shadow-theme-md">
              <p className="text-sm font-semibold">OmniRAG Console</p>
              <p className="mt-2 text-sm text-white/80">
                {locale === "zh"
                  ? "在一个工作区中完成文本、图片、视频知识的导入、检索与治理。"
                  : "Search, ingest and manage text, image and video knowledge in one workspace."}
              </p>
            </div>
          </div>
        ) : null}
      </div>
    </aside>
  );
};

export default AppSidebar;
