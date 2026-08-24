import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";

import LandingPage from "./pages/LandingPage";
import ForecastDashboard from "./pages/ForecastDashboard";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={<LandingPage />}
        />

        <Route
          path="/forecast"
          element={<ForecastDashboard />}
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;