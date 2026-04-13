import { BrowserRouter as Router, Navigate, Route, Routes } from "react-router";
import AppLayout from "./layout/AppLayout";
import { ScrollToTop } from "./components/common/ScrollToTop";
import OverviewPage from "./pages/OmniRAG/OverviewPage";
import SettingsPage from "./pages/OmniRAG/SettingsPage";
import IngestPage from "./pages/OmniRAG/IngestPage";
import SearchPage from "./pages/OmniRAG/SearchPage";
import HybridSearchPage from "./pages/OmniRAG/HybridSearchPage";
import RecordsPage from "./pages/OmniRAG/RecordsPage";
import NotFound from "./pages/OtherPage/NotFound";
import { OmniRAGProvider } from "./context/OmniRAGContext";

export default function App() {
  return (
    <Router>
      <OmniRAGProvider>
        <ScrollToTop />
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<OverviewPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/ingest" element={<IngestPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/hybrid-search" element={<HybridSearchPage />} />
            <Route path="/records" element={<RecordsPage />} />
            <Route path="/home" element={<Navigate to="/" replace />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </OmniRAGProvider>
    </Router>
  );
}
