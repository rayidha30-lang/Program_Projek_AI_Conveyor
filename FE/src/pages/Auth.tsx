import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Eye,
  EyeOff,
  Cpu,
  CheckCircle2,
  AlertCircle,
  Video,
  ShieldCheck,
  Zap,
  Activity,
  ArrowRight,
} from "lucide-react";

export default function Auth() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [showPassword, setShowPassword] = useState(false);

  // Form states
  const [fullName, setFullName] = useState("");
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin");
  const [rememberMe, setRememberMe] = useState(true);
  const [agreeTerms, setAgreeTerms] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage("");

    if (mode === "login") {
      const u = username.trim().toLowerCase();
      const p = password.trim();
      const registeredPw = localStorage.getItem(`user_${u}`);

      // Validasi akun internal: default admin/admin atau akun registrasi lokal
      if ((u === "admin" && p === "admin") || (registeredPw && registeredPw === p)) {
        setIsLoading(true);
        setTimeout(() => {
          localStorage.setItem("isAuthenticated", "true");
          localStorage.setItem(
            "userName",
            u === "admin" ? "Admin (QA Lead)" : username.trim()
          );
          setIsLoading(false);
          navigate("/");
        }, 350);
      } else {
        setErrorMessage("Username atau password salah. Default akun internal: admin / admin");
      }
    } else {
      // Mode Register Akun Internal
      const u = username.trim().toLowerCase();
      if (!u || !password.trim()) {
        setErrorMessage("Silakan lengkapi username dan password.");
        return;
      }
      setIsLoading(true);
      setTimeout(() => {
        localStorage.setItem(`user_${u}`, password.trim());
        localStorage.setItem("isAuthenticated", "true");
        localStorage.setItem("userName", fullName.trim() || username.trim());
        setIsLoading(false);
        navigate("/");
      }, 350);
    }
  };

  return (
    <div className="min-h-screen bg-[#eaedf2] flex items-center justify-center p-3 sm:p-5 lg:p-6 font-sans antialiased text-neutral-800 selection:bg-purple-200">
      {/* Outer Card enclosing both Auth form and Showcase Banner */}
      <main className="w-full max-w-[1240px] min-h-[680px] lg:h-[720px] bg-white rounded-[32px] shadow-2xl shadow-slate-300/70 overflow-hidden flex flex-col lg:flex-row relative border border-slate-200/60">
        {/* Sisi Kiri: Panel Autentikasi (Form Login / Register) */}
        <section
          className="w-full lg:w-[50%] flex flex-col justify-between p-6 sm:p-10 lg:p-12 xl:p-14 relative"
          data-purpose="auth-form-panel"
        >
          {/* Header Mobile Brand */}
          <div className="flex items-center gap-2.5 lg:hidden mb-6">
            <div className="flex size-9 items-center justify-center rounded-xl bg-linear-to-br from-[#7645f7] to-[#5a24e2] text-white shadow-md">
              <Cpu className="size-5" />
            </div>
            <div>
              <p className="text-xs font-black tracking-tight text-neutral-900">
                Die Cast Machine #2
              </p>
              <p className="text-[9px] font-bold tracking-wider text-purple-600">
                SLAT RIB VISION SYSTEM
              </p>
            </div>
          </div>

          <div className="w-full max-w-[390px] mx-auto my-auto flex flex-col">
            {/* Header Titles */}
            <div className="mb-5">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-purple-50 px-3 py-1 text-[11px] font-bold text-purple-700 ring-1 ring-purple-500/20 mb-2.5">
                <ShieldCheck className="size-3.5" />
                Internal Machine & Plant Portal
              </span>
              <h1 className="text-[28px] sm:text-[30px] font-black tracking-tight text-neutral-900 leading-tight">
                {mode === "login" ? "Welcome back" : "Create account"}
              </h1>
              <p className="text-[13px] text-neutral-500 font-normal mt-1 leading-relaxed">
                {mode === "login"
                  ? "Enter internal operator credentials to access the AI Vision Inspection Dashboard."
                  : "Register QA / Maintenance personnel for internal machine inspection access."}
              </p>
            </div>

            {/* Default Internal Credentials Helper Card */}
            {mode === "login" && (
              <div className="mb-4 rounded-2xl bg-purple-50/70 border border-purple-200/80 p-3 text-xs text-purple-900 flex items-center justify-between shadow-xs">
                <div className="flex items-center gap-2.5">
                  <div className="size-8 rounded-xl bg-linear-to-br from-[#7645f7] to-[#5a24e2] text-white flex items-center justify-center font-black text-[10px] shadow-xs">
                    PIC
                  </div>
                  <div>
                    <span className="block text-[9.5px] font-extrabold uppercase tracking-wider text-purple-600">
                      Internal Credentials
                    </span>
                    <span className="font-mono text-[11px] font-black text-purple-950">
                      Username: admin · Password: admin
                    </span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setUsername("admin");
                    setPassword("admin");
                    setErrorMessage("");
                  }}
                  className="rounded-lg bg-purple-600 hover:bg-purple-700 text-white px-2.5 py-1 text-[10.5px] font-bold active:scale-95 transition cursor-pointer shadow-xs"
                >
                  Auto Fill
                </button>
              </div>
            )}

            {errorMessage && (
              <div className="mb-4 rounded-xl bg-rose-50 border border-rose-200 p-3 text-xs font-semibold text-rose-700 flex items-center gap-2">
                <AlertCircle className="size-4 shrink-0 text-rose-600" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Form Elements */}
            <form onSubmit={handleSubmit} className="space-y-3.5">
              {mode === "register" && (
                <div className="space-y-1">
                  <label
                    htmlFor="fullName"
                    className="block text-xs font-semibold text-neutral-800"
                  >
                    Full Name & Title
                  </label>
                  <input
                    id="fullName"
                    type="text"
                    required
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="e.g. Sarah Jenkins (QA Lead)"
                    className="w-full h-10 px-3.5 text-xs text-neutral-800 rounded-full border border-neutral-200 focus:border-[#6a37eb] focus:ring-2 focus:ring-[#6a37eb]/20 outline-none transition placeholder:text-neutral-400"
                  />
                </div>
              )}

              {/* Username Input */}
              <div className="space-y-1">
                <label
                  htmlFor="username"
                  className="block text-xs font-semibold text-neutral-800"
                >
                  {mode === "register" ? "Operator Username" : "Username"}
                </label>
                <input
                  id="username"
                  type="text"
                  required
                  value={username}
                  onChange={(e) => {
                    setUsername(e.target.value);
                    if (errorMessage) setErrorMessage("");
                  }}
                  placeholder="e.g. admin"
                  className="w-full h-10 px-3.5 text-xs text-neutral-800 rounded-full border border-neutral-200 focus:border-[#6a37eb] focus:ring-2 focus:ring-[#6a37eb]/20 outline-none transition placeholder:text-neutral-400"
                />
              </div>

              {/* Password Input */}
              <div className="space-y-1">
                <label
                  htmlFor="password"
                  className="block text-xs font-semibold text-neutral-800"
                >
                  Password
                </label>
                <div className="relative">
                  <input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    required
                    value={password}
                    onChange={(e) => {
                      setPassword(e.target.value);
                      if (errorMessage) setErrorMessage("");
                    }}
                    placeholder={
                      mode === "register"
                        ? "Create password (min. 4 chars)"
                        : "Enter password (e.g. admin)"
                    }
                    className="w-full h-10 pl-3.5 pr-10 text-xs text-neutral-800 rounded-full border border-neutral-200 focus:border-[#6a37eb] focus:ring-2 focus:ring-[#6a37eb]/20 outline-none transition placeholder:text-neutral-400"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-neutral-600 transition p-1 cursor-pointer"
                    title={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? (
                      <EyeOff className="size-4" />
                    ) : (
                      <Eye className="size-4" />
                    )}
                  </button>
                </div>
                {mode === "register" && (
                  <p className="text-[10px] text-neutral-400 mt-1 pl-1">
                    Internal password format (min. 4 characters)
                  </p>
                )}
              </div>

              {/* Checkboxes & Links */}
              {mode === "login" ? (
                <div className="flex items-center justify-between pt-0.5">
                  <label className="flex items-center gap-2 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={rememberMe}
                      onChange={(e) => setRememberMe(e.target.checked)}
                      className="size-3.5 rounded text-[#6a37eb] border-neutral-300 focus:ring-[#6a37eb] cursor-pointer"
                    />
                    <span className="text-xs text-neutral-700 font-medium">
                      Remember me
                    </span>
                  </label>
                  <a
                    href="#forgot"
                    onClick={(e) => {
                      e.preventDefault();
                      alert("Akses Internal: Default kredensial sistem adalah Username: admin, Password: admin. Silakan hubungi supervisor mesin jika memerlukan bantuan akses.");
                    }}
                    className="text-xs font-semibold text-[#6a37eb] hover:text-[#5a24e2] transition"
                  >
                    Forgot password?
                  </a>
                </div>
              ) : (
                <div className="flex items-start gap-2 pt-0.5">
                  <input
                    id="terms"
                    type="checkbox"
                    required
                    checked={agreeTerms}
                    onChange={(e) => setAgreeTerms(e.target.checked)}
                    className="size-3.5 mt-0.5 rounded text-[#6a37eb] border-neutral-300 focus:ring-[#6a37eb] cursor-pointer"
                  />
                  <label
                    htmlFor="terms"
                    className="text-xs text-neutral-600 font-normal leading-tight cursor-pointer select-none"
                  >
                    I agree to the{" "}
                    <span className="font-semibold text-[#6a37eb]">
                      Safety & Operational Protocols
                    </span>{" "}
                    and{" "}
                    <span className="font-semibold text-[#6a37eb]">
                      Traceability Policy
                    </span>
                  </label>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full h-11 mt-2 bg-linear-to-r from-[#7645f7] via-[#6a37eb] to-[#5a24e2] hover:opacity-95 text-white text-xs font-bold rounded-full shadow-lg shadow-purple-600/25 transition duration-150 flex items-center justify-center gap-2 tracking-wide cursor-pointer disabled:opacity-50"
              >
                {isLoading ? (
                  <span className="flex items-center gap-2">
                    <span className="size-3.5 animate-spin rounded-full border-2 border-white border-t-transparent"></span>
                    Authenticating...
                  </span>
                ) : (
                  <>
                    <span>{mode === "login" ? "Sign in to Dashboard" : "Complete Registration"}</span>
                    <ArrowRight className="size-4" />
                  </>
                )}
              </button>
            </form>

            {/* Toggle Mode Link */}
            <p className="text-center text-xs text-neutral-600 mt-5">
              {mode === "login" ? (
                <>
                  Don't have an operator account?{" "}
                  <button
                    onClick={() => setMode("register")}
                    className="font-bold text-[#6a37eb] hover:text-[#5a24e2] transition cursor-pointer underline underline-offset-2"
                  >
                    Register
                  </button>
                </>
              ) : (
                <>
                  Already registered?{" "}
                  <button
                    onClick={() => setMode("login")}
                    className="font-bold text-[#6a37eb] hover:text-[#5a24e2] transition cursor-pointer underline underline-offset-2"
                  >
                    Sign in
                  </button>
                </>
              )}
            </p>
          </div>

          {/* Copyright & Machine Info Footer */}
          <footer className="text-center pt-6 text-[11px] text-neutral-400 font-normal">
            © 2026 Die Cast Machine #2 · Slat Rib Conveyor AI Vision System
          </footer>
        </section>

        {/* Sisi Kanan: Panel Showcase Ungu (Brand & Mini Vision Dashboard Preview) */}
        <section className="w-full lg:w-[50%] bg-linear-to-br from-[#7645f7] via-[#6a37eb] to-[#5a24e2] p-6 sm:p-8 lg:p-10 text-white flex flex-col justify-between relative overflow-hidden shadow-inner">
          {/* Top Branding Section */}
          <div className="relative z-10 shrink-0">
            <div className="flex items-center gap-3 mb-4">
              <div className="size-9 rounded-2xl bg-white/20 backdrop-blur-md flex items-center justify-center shadow-md">
                <Cpu className="size-5 text-white" />
              </div>
              <div>
                <span className="text-base font-extrabold tracking-tight text-white block leading-tight">
                  Die Cast Machine #2
                </span>
                <span className="text-[10px] font-bold tracking-wider text-purple-200">
                  SLAT RIB VISION SYSTEM
                </span>
              </div>
            </div>

            <h2 className="text-2xl sm:text-[27px] lg:text-[29px] font-black leading-tight max-w-[440px] tracking-tight">
              Continuous AI Optical Defect Surveillance
            </h2>
            <p className="text-purple-100/90 text-xs sm:text-[13px] leading-relaxed max-w-[450px] mt-2 font-light">
              High-speed visual inspection for Plat/Slat, Rib, Bolt & Pin abnormalities with microsecond inference and verifiable maintenance ledger.
            </p>
          </div>

          {/* Embedded Live Inspection Mockup - Dinaikkan sesuai panah & difullkan sampai ke bawah */}
          <div className="mt-4 lg:mt-5 flex-1 flex flex-col justify-between relative z-10 rounded-2xl bg-[#f8fafc] text-neutral-800 p-4 sm:p-5 shadow-2xl border border-white/20 overflow-hidden transform transition hover:scale-[1.005] duration-300 min-h-0">
            {/* Mockup Top Header */}
            <div className="flex items-center justify-between pb-3 border-b border-neutral-100 shrink-0">
              <div className="flex items-center gap-2">
                <span className="flex size-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="text-[11px] font-black text-neutral-900 tracking-wide">
                  INSPECTION TELEMETRY ACTIVE
                </span>
              </div>
              <div className="flex items-center gap-2 font-mono text-[9.5px] text-neutral-500 font-semibold">
                <Activity className="size-3.5 text-[#6a37eb]" />
                <span>3 Cams Synchronized</span>
              </div>
            </div>

            {/* 3 Cameras Preview Mini Grid - Expanded height to fill space */}
            <div className="grid grid-cols-3 gap-2.5 my-3 flex-1 min-h-0">
              <div className="rounded-xl bg-neutral-900 p-2.5 text-white flex flex-col justify-between relative overflow-hidden border border-neutral-700 shadow-sm">
                <div className="flex items-center justify-between text-[7.5px] text-neutral-300 shrink-0">
                  <span className="font-bold text-white flex items-center gap-1">
                    <Video className="size-2.5 text-emerald-400" /> CAM-01
                  </span>
                  <span className="text-emerald-400 font-mono font-bold">30 FPS</span>
                </div>
                <div className="flex flex-col items-center justify-center my-auto py-1">
                  <span className="text-[9.5px] font-extrabold text-purple-200">Plat & Rib</span>
                  <span className="text-[7.5px] text-neutral-400 font-medium">Overhead Area</span>
                </div>
                <div className="flex justify-between items-center text-[7px] text-neutral-400 font-mono border-t border-neutral-800 pt-1 shrink-0">
                  <span>Conf: 98.4%</span>
                  <span className="text-emerald-400 font-bold bg-emerald-950/70 px-1 py-0.5 rounded text-[6.5px]">NORMAL</span>
                </div>
              </div>

              <div className="rounded-xl bg-neutral-900 p-2.5 text-white flex flex-col justify-between relative overflow-hidden border border-neutral-700 shadow-sm">
                <div className="flex items-center justify-between text-[7.5px] text-neutral-300 shrink-0">
                  <span className="font-bold text-white flex items-center gap-1">
                    <Video className="size-2.5 text-emerald-400" /> CAM-02
                  </span>
                  <span className="text-emerald-400 font-mono font-bold">30 FPS</span>
                </div>
                <div className="flex flex-col items-center justify-center my-auto py-1">
                  <span className="text-[9.5px] font-extrabold text-blue-200">Bolt & Pin RH</span>
                  <span className="text-[7.5px] text-neutral-400 font-medium">Right Oblique</span>
                </div>
                <div className="flex justify-between items-center text-[7px] text-neutral-400 font-mono border-t border-neutral-800 pt-1 shrink-0">
                  <span>Conf: 97.2%</span>
                  <span className="text-emerald-400 font-bold bg-emerald-950/70 px-1 py-0.5 rounded text-[6.5px]">NORMAL</span>
                </div>
              </div>

              <div className="rounded-xl bg-neutral-900 p-2.5 text-white flex flex-col justify-between relative overflow-hidden border border-neutral-700 shadow-sm">
                <div className="flex items-center justify-between text-[7.5px] text-neutral-300 shrink-0">
                  <span className="font-bold text-white flex items-center gap-1">
                    <Video className="size-2.5 text-emerald-400" /> CAM-03
                  </span>
                  <span className="text-emerald-400 font-mono font-bold">30 FPS</span>
                </div>
                <div className="flex flex-col items-center justify-center my-auto py-1">
                  <span className="text-[9.5px] font-extrabold text-rose-200">Bolt & Pin LH</span>
                  <span className="text-[7.5px] text-neutral-400 font-medium">Left Carriage</span>
                </div>
                <div className="flex justify-between items-center text-[7px] text-neutral-400 font-mono border-t border-neutral-800 pt-1 shrink-0">
                  <span>Conf: 99.1%</span>
                  <span className="text-emerald-400 font-bold bg-emerald-950/70 px-1 py-0.5 rounded text-[6.5px]">NORMAL</span>
                </div>
              </div>
            </div>

            {/* Mini Metrics Row */}
            <div className="grid grid-cols-3 gap-2 shrink-0">
              <div className="rounded-xl bg-neutral-100 p-2 text-neutral-800 border border-neutral-200/50">
                <span className="text-[8px] text-neutral-500 block font-semibold">Total Inspected</span>
                <span className="text-[13px] font-black block text-neutral-900 leading-tight mt-0.5">12,840</span>
              </div>
              <div className="rounded-xl bg-emerald-50 border border-emerald-200/70 p-2 text-emerald-900">
                <span className="text-[8px] text-emerald-700 block font-semibold">Normal Pass</span>
                <span className="text-[13px] font-black block text-emerald-700 leading-tight mt-0.5">12,817</span>
              </div>
              <div className="rounded-xl bg-rose-50 border border-rose-200/70 p-2 text-rose-900">
                <span className="text-[8px] text-rose-700 block font-semibold">Abnormal Flag</span>
                <span className="text-[13px] font-black block text-rose-700 leading-tight mt-0.5">23 (0.18%)</span>
              </div>
            </div>

            {/* Bottom Feature Badges */}
            <div className="mt-3 pt-2.5 border-t border-neutral-100 flex items-center justify-between text-[8.5px] font-bold text-neutral-600 shrink-0">
              <span className="flex items-center gap-1 text-emerald-700">
                <CheckCircle2 className="size-3.5 text-emerald-600" /> Real-time Inference
              </span>
              <span className="flex items-center gap-1 text-amber-700">
                <Zap className="size-3.5 text-amber-600" /> 16ms Latency
              </span>
              <span className="flex items-center gap-1 text-purple-700">
                <ShieldCheck className="size-3.5 text-[#6a37eb]" /> Audit Ledger
              </span>
            </div>
          </div>

          {/* Ambient Background Glow Effect */}
          <div className="absolute -bottom-20 -right-20 size-72 bg-purple-400/25 rounded-full blur-3xl pointer-events-none"></div>
        </section>
      </main>
    </div>
  );
}
