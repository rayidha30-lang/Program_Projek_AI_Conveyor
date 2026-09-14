import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Eye,
  EyeOff,
  Cpu,
  CheckCircle2,
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
  const [email, setEmail] = useState("s.jenkins@company.com");
  const [password, setPassword] = useState("••••••••••••");
  const [rememberMe, setRememberMe] = useState(true);
  const [agreeTerms, setAgreeTerms] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    setTimeout(() => {
      localStorage.setItem("isAuthenticated", "true");
      if (mode === "register" && fullName.trim()) {
        localStorage.setItem("userName", fullName.trim());
      }
      setIsLoading(false);
      navigate("/");
    }, 450);
  };

  const handleGoogleAuth = () => {
    setIsLoading(true);
    setTimeout(() => {
      localStorage.setItem("isAuthenticated", "true");
      setIsLoading(false);
      navigate("/");
    }, 450);
  };

  return (
    <div className="min-h-screen bg-[#eaedf2] flex items-center justify-center p-3 sm:p-6 lg:p-8 font-sans antialiased text-neutral-800 selection:bg-purple-200">
      {/* Outer Card enclosing both Auth form and Showcase Banner */}
      <main className="w-full max-w-[1240px] min-h-[760px] bg-white rounded-[32px] shadow-2xl shadow-slate-300/70 overflow-hidden flex flex-col lg:flex-row relative border border-slate-200/60">
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
            <div className="mb-6">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-purple-50 px-3 py-1 text-[11px] font-bold text-purple-700 ring-1 ring-purple-500/20 mb-3">
                <ShieldCheck className="size-3.5" />
                Authorized QA & PIC Portal
              </span>
              <h1 className="text-[30px] font-black tracking-tight text-neutral-900 leading-tight">
                {mode === "login" ? "Welcome back" : "Create account"}
              </h1>
              <p className="text-[13px] text-neutral-500 font-normal mt-1.5 leading-relaxed">
                {mode === "login"
                  ? "Enter your credentials to access the AI Vision Inspection Dashboard."
                  : "Register QA / Maintenance personnel to access machine telemetry."}
              </p>
            </div>

            {/* Google OAuth Button */}
            <button
              onClick={handleGoogleAuth}
              disabled={isLoading}
              type="button"
              className="w-full h-11 flex items-center justify-center gap-2.5 rounded-full border border-neutral-200 bg-white hover:bg-neutral-50 active:bg-neutral-100 transition duration-150 text-neutral-700 text-xs font-semibold shadow-xs cursor-pointer"
            >
              <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24">
                <path
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  fill="#4285F4"
                />
                <path
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  fill="#34A853"
                />
                <path
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                  fill="#FBBC05"
                />
                <path
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                  fill="#EA4335"
                />
              </svg>
              <span>
                {mode === "login"
                  ? "Sign in with Google Workspace"
                  : "Sign up with Google Workspace"}
              </span>
            </button>

            {/* Divider */}
            <div className="relative flex py-4 items-center">
              <div className="flex-grow border-t border-neutral-200"></div>
              <span className="flex-shrink mx-3 text-neutral-400 text-xs lowercase">
                or continue with email
              </span>
              <div className="flex-grow border-t border-neutral-200"></div>
            </div>

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

              {/* Email Input */}
              <div className="space-y-1">
                <label
                  htmlFor="email"
                  className="block text-xs font-semibold text-neutral-800"
                >
                  {mode === "register" ? "Work Email" : "Email Address"}
                </label>
                <input
                  id="email"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@company.com"
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
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={
                      mode === "register"
                        ? "Create password (min. 8 chars)"
                        : "Enter password"
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
                    Must be at least 8 characters with letters & numbers
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
                      alert("Please contact system admin (diecast-admin@company.com) to reset password.");
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
        <section className="w-full lg:w-[50%] p-3 lg:p-4 flex">
          <div className="w-full bg-linear-to-br from-[#7645f7] via-[#6a37eb] to-[#5a24e2] rounded-[26px] p-6 sm:p-8 lg:p-10 text-white flex flex-col justify-between overflow-hidden relative shadow-inner">
            {/* Top Branding Section */}
            <div className="relative z-10">
              <div className="flex items-center gap-3 mb-5">
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

              <h2 className="text-2xl sm:text-[28px] lg:text-[30px] font-black leading-tight max-w-[430px] tracking-tight">
                Continuous AI Optical Defect Surveillance
              </h2>
              <p className="text-purple-100/90 text-xs sm:text-[13px] leading-relaxed max-w-[440px] mt-2.5 font-light">
                High-speed visual inspection for Plat/Slat, Rib, Bolt & Pin abnormalities with microsecond inference and verifiable maintenance ledger.
              </p>
            </div>

            {/* Embedded Live Inspection Mockup */}
            <div className="mt-6 relative z-10 rounded-2xl bg-[#f8fafc] text-neutral-800 p-3.5 shadow-2xl border border-white/20 overflow-hidden transform transition hover:scale-[1.01] duration-300">
              {/* Mockup Top Header */}
              <div className="flex items-center justify-between pb-2.5 border-b border-neutral-100">
                <div className="flex items-center gap-2">
                  <span className="flex size-2 rounded-full bg-emerald-500 animate-pulse"></span>
                  <span className="text-[10px] font-black text-neutral-900 tracking-wide">
                    INSPECTION TELEMETRY ACTIVE
                  </span>
                </div>
                <div className="flex items-center gap-2 font-mono text-[9px] text-neutral-500 font-semibold">
                  <Activity className="size-3 text-[#6a37eb]" />
                  <span>3 Cams Synchronized</span>
                </div>
              </div>

              {/* 3 Cameras Preview Mini Grid */}
              <div className="grid grid-cols-3 gap-1.5 mt-2.5">
                <div className="rounded-lg bg-neutral-900 p-1.5 text-white flex flex-col justify-between aspect-4/3 relative overflow-hidden border border-neutral-700">
                  <div className="flex items-center justify-between text-[7px] text-neutral-300">
                    <span className="font-bold text-white flex items-center gap-1">
                      <Video className="size-2 text-emerald-400" /> CAM-01
                    </span>
                    <span className="text-emerald-400 font-mono">30 FPS</span>
                  </div>
                  <div className="flex items-center justify-center my-auto">
                    <span className="text-[8px] font-semibold text-purple-200">Plat & Rib</span>
                  </div>
                  <div className="flex justify-between text-[6.5px] text-neutral-400 font-mono">
                    <span>Conf: 98.4%</span>
                    <span className="text-emerald-400">NORMAL</span>
                  </div>
                </div>

                <div className="rounded-lg bg-neutral-900 p-1.5 text-white flex flex-col justify-between aspect-4/3 relative overflow-hidden border border-neutral-700">
                  <div className="flex items-center justify-between text-[7px] text-neutral-300">
                    <span className="font-bold text-white flex items-center gap-1">
                      <Video className="size-2 text-emerald-400" /> CAM-02
                    </span>
                    <span className="text-emerald-400 font-mono">30 FPS</span>
                  </div>
                  <div className="flex items-center justify-center my-auto">
                    <span className="text-[8px] font-semibold text-blue-200">Bolt & Pin RH</span>
                  </div>
                  <div className="flex justify-between text-[6.5px] text-neutral-400 font-mono">
                    <span>Conf: 97.2%</span>
                    <span className="text-emerald-400">NORMAL</span>
                  </div>
                </div>

                <div className="rounded-lg bg-neutral-900 p-1.5 text-white flex flex-col justify-between aspect-4/3 relative overflow-hidden border border-neutral-700">
                  <div className="flex items-center justify-between text-[7px] text-neutral-300">
                    <span className="font-bold text-white flex items-center gap-1">
                      <Video className="size-2 text-emerald-400" /> CAM-03
                    </span>
                    <span className="text-emerald-400 font-mono">30 FPS</span>
                  </div>
                  <div className="flex items-center justify-center my-auto">
                    <span className="text-[8px] font-semibold text-rose-200">Bolt & Pin LH</span>
                  </div>
                  <div className="flex justify-between text-[6.5px] text-neutral-400 font-mono">
                    <span>Conf: 99.1%</span>
                    <span className="text-emerald-400">NORMAL</span>
                  </div>
                </div>
              </div>

              {/* Mini Metrics Row */}
              <div className="grid grid-cols-3 gap-1.5 mt-2.5">
                <div className="rounded-lg bg-neutral-100 p-1.5 text-neutral-800">
                  <span className="text-[7px] text-neutral-400 block font-semibold">Total Inspected</span>
                  <span className="text-[11px] font-extrabold block">12,840</span>
                </div>
                <div className="rounded-lg bg-emerald-50 border border-emerald-200/60 p-1.5 text-emerald-800">
                  <span className="text-[7px] text-emerald-600 block font-semibold">Normal Pass</span>
                  <span className="text-[11px] font-extrabold block">12,817</span>
                </div>
                <div className="rounded-lg bg-rose-50 border border-rose-200/60 p-1.5 text-rose-800">
                  <span className="text-[7px] text-rose-600 block font-semibold">Abnormal Flag</span>
                  <span className="text-[11px] font-extrabold block">23 (0.18%)</span>
                </div>
              </div>

              {/* Bottom Feature Badges */}
              <div className="mt-2.5 pt-2 border-t border-neutral-100 flex items-center justify-between text-[8px] font-bold text-neutral-600">
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="size-3 text-emerald-600" /> Real-time Inference
                </span>
                <span className="flex items-center gap-1">
                  <Zap className="size-3 text-amber-500" /> 16ms Latency
                </span>
                <span className="flex items-center gap-1">
                  <ShieldCheck className="size-3 text-[#6a37eb]" /> Audit Ledger
                </span>
              </div>
            </div>

            {/* Ambient Background Glow Effect */}
            <div className="absolute -bottom-20 -right-20 size-72 bg-purple-400/25 rounded-full blur-3xl pointer-events-none"></div>
          </div>
        </section>
      </main>
    </div>
  );
}
