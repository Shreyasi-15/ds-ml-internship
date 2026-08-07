const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ??
  "http://localhost:8000"
).replace(/\/$/, "");

export type AuthUser = {
  id: number;
  email: string;
  display_name: string;
  created_at: string;
};

type RegisterData = {
  display_name: string;
  email: string;
  password: string;
};

type LoginData = {
  email: string;
  password: string;
};

async function authRequest<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    {
      ...options,
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    },
  );

  if (!response.ok) {
    let message = "The request could not be completed.";

    try {
      const body = (await response.json()) as {
        detail?: string;
      };

      if (body.detail) {
        message = body.detail;
      }
    } catch {
      message = `Server returned status ${response.status}.`;
    }

    throw new Error(message);
  }

  return (await response.json()) as T;
}

export function registerAccount(
  data: RegisterData,
): Promise<AuthUser> {
  return authRequest<AuthUser>("/auth/register", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function loginAccount(
  data: LoginData,
): Promise<AuthUser> {
  return authRequest<AuthUser>("/auth/login", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getCurrentUser():
Promise<AuthUser | null> {
  const response = await fetch(
    `${API_BASE_URL}/auth/me`,
    {
      credentials: "include",
    },
  );

  if (response.status === 401) {
    return null;
  }

  if (!response.ok) {
    throw new Error(
      "The current account could not be loaded.",
    );
  }

  return (await response.json()) as AuthUser;
}

export async function logoutAccount(): Promise<void> {
  await authRequest<{ message: string }>(
    "/auth/logout",
    {
      method: "POST",
    },
  );
}