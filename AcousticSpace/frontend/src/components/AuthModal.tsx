import { useState } from "react";
import type { FormEvent } from "react";
import {
  Eye,
  EyeOff,
  LogIn,
  LogOut,
  ShieldCheck,
  X,
} from "lucide-react";

import {
  loginAccount,
  logoutAccount,
  registerAccount,
} from "../api/auth";
import type { AuthUser } from "../api/auth";

type AuthModalProps = {
  user: AuthUser | null;
  onAuthenticated: (user: AuthUser) => void;
  onLogout: () => void;
  onClose: () => void;
};

export function AuthModal({
  user,
  onAuthenticated,
  onLogout,
  onClose,
}: AuthModalProps) {
  const [mode, setMode] =
    useState<"login" | "register">("login");
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] =
  useState("");
  const [showPassword, setShowPassword] =
  useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (
    event: FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();
    if (
  mode === "register" &&
  password !== confirmPassword
) {
  setError("The passwords do not match.");
  return;
}
    setSubmitting(true);
    setError(null);

    try {
      const authenticatedUser =
        mode === "register"
          ? await registerAccount({
              display_name: displayName,
              email,
              password,
            })
          : await loginAccount({
              email,
              password,
            });

      onAuthenticated(authenticatedUser);
      onClose();
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Authentication failed.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleLogout = async () => {
    setSubmitting(true);
    setError(null);

    try {
      await logoutAccount();
      onLogout();
      onClose();
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Sign out failed.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="auth-backdrop"
      role="presentation"
      onMouseDown={onClose}
    >
      <section
        className="auth-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button
          type="button"
          className="auth-close"
          aria-label="Close account window"
          onClick={onClose}
        >
          <X size={20} />
        </button>

        <div className="auth-brand">
          <ShieldCheck size={27} />
        </div>

        {user ? (
          <>
            <h2 id="auth-title">Your account</h2>
            <p className="auth-subtitle">
              You are signed in to AcousticSpace.
            </p>

            <div className="account-details">
              <span>Display name</span>
              <strong>{user.display_name}</strong>

              <span>Email address</span>
              <strong>{user.email}</strong>
            </div>

            {error && (
              <p className="auth-error" role="alert">
                {error}
              </p>
            )}

            <button
              type="button"
              className="auth-submit logout-button"
              disabled={submitting}
              onClick={handleLogout}
            >
              <LogOut size={18} />
              {submitting ? "Signing out..." : "Sign out"}
            </button>
          </>
        ) : (
          <>
            <h2 id="auth-title">
              {mode === "login"
                ? "Welcome back"
                : "Create analyst account"}
            </h2>

            <p className="auth-subtitle">
              {mode === "login"
                ? "Sign in to analyze audio recordings."
                : "Register securely using your email address."}
            </p>

            <div className="auth-tabs">
              <button
                type="button"
                className={mode === "login" ? "active" : ""}
                onClick={() => {
                  setMode("login");
                  setError(null);
                }}
              >
                Sign in
              </button>

              <button
                type="button"
                className={mode === "register" ? "active" : ""}
                onClick={() => {
                  setMode("register");
                  setError(null);
                }}
              >
                Register
              </button>
            </div>

            <form className="auth-form" onSubmit={handleSubmit}>
              {mode === "register" && (
                <label>
                  Display name
                  <input
                    required
                    minLength={2}
                    maxLength={60}
                    value={displayName}
                    autoComplete="name"
                    placeholder="Shreyasi Chowdhury"
                    onChange={(event) =>
                      setDisplayName(event.target.value)
                    }
                  />
                </label>
              )}

              <label>
                Email address
                <input
                  required
                  type="email"
                  maxLength={254}
                  value={email}
                  autoComplete="email"
                  placeholder="name@example.com"
                  onChange={(event) =>
                    setEmail(event.target.value)
                  }
                />
              </label>

              <label>
  Password

  <div className="password-input">
    <input
      required
      type={showPassword ? "text" : "password"}
      minLength={mode === "register" ? 8 : 1}
      maxLength={128}
      value={password}
      autoComplete={
        mode === "register"
          ? "new-password"
          : "current-password"
      }
      placeholder="Enter your password"
      onChange={(event) =>
        setPassword(event.target.value)
      }
    />

    <button
      type="button"
      aria-label={
        showPassword
          ? "Hide password"
          : "Show password"
      }
      onClick={() =>
        setShowPassword((visible) => !visible)
      }
    >
      {showPassword ? (
        <EyeOff size={18} />
      ) : (
        <Eye size={18} />
      )}
    </button>
  </div>
</label>
{mode === "register" && (
  <label>
    Confirm password

    <div className="password-input">
      <input
        required
        type={showPassword ? "text" : "password"}
        minLength={8}
        maxLength={128}
        value={confirmPassword}
        autoComplete="new-password"
        placeholder="Enter the password again"
        onChange={(event) =>
          setConfirmPassword(event.target.value)
        }
      />

      <button
        type="button"
        aria-label={
          showPassword
            ? "Hide passwords"
            : "Show passwords"
        }
        onClick={() =>
          setShowPassword((visible) => !visible)
        }
      >
        {showPassword ? (
          <EyeOff size={18} />
        ) : (
          <Eye size={18} />
        )}
      </button>
    </div>
  </label>
)}
              {mode === "register" && (
                <p className="password-guidance">
                  Use at least 8 characters with one letter
                  and one number.
                </p>
              )}

              {error && (
                <p className="auth-error" role="alert">
                  {error}
                </p>
              )}

              <button
                type="submit"
                className="auth-submit"
                disabled={submitting}
              >
                <LogIn size={18} />

                {submitting
                  ? "Please wait..."
                  : mode === "login"
                    ? "Sign in"
                    : "Create account"}
              </button>
            </form>
          </>
        )}
      </section>
    </div>
  );
}