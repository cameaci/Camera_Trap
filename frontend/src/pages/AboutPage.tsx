/**
 * About page.
 *
 * App-level page (not project-scoped). Reachable from the global
 * hamburger menu. Reads the version via the Electron IPC; in dev /
 * browser the version falls back to "(dev)".
 *
 * The contributors row pulls from GitHub's public contributors API at
 * page-open time. If GitHub is unreachable or rate-limited, the
 * component degrades silently — the surrounding copy still makes sense
 * without avatars.
 */

import { useEffect, useState } from "react";
import { HomeButton } from "../components/layout/HomeButton";
import { useQuery } from "@tanstack/react-query";
import { Tag } from "lucide-react";
import { formatVersion } from "@/lib/version";
import { ISSUES_URL, PRODUCT_NAME, PROJECT_URL, REPO } from "@/lib/wsp";

const LICENSE_URL = `${PROJECT_URL}/blob/main/LICENSE`;

interface GithubContributor {
  login: string;
  id: number;
  avatar_url: string;
  html_url: string;
  contributions: number;
  type: string;
}

export default function AboutPage() {
  const [version, setVersion] = useState<string>("(dev)");

  useEffect(() => {
    if (typeof window !== "undefined" && window.electronAPI?.getVersion) {
      window.electronAPI.getVersion().then(setVersion).catch(() => {
        setVersion("(unknown)");
      });
    }
  }, []);

  const { data: contributors } = useQuery({
    queryKey: ["contributors", REPO],
    queryFn: async (): Promise<GithubContributor[]> => {
      const res = await fetch(
        `https://api.github.com/repos/${REPO}/contributors?per_page=30`,
      );
      if (!res.ok) {
        throw new Error(`GitHub returned ${res.status}`);
      }
      const all: GithubContributor[] = await res.json();
      // Drop obvious bots (dependabot, github-actions[bot], etc.).
      return all.filter(
        (c) => c.type !== "Bot" && !c.login.toLowerCase().includes("[bot]"),
      );
    },
    staleTime: 60 * 60 * 1000, // 1 hour
    retry: false,
  });

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <HomeButton />
            <img
              src="/branding/logo-mark.png"
              alt={PRODUCT_NAME}
              className="h-12 w-12 shrink-0"
            />
            <div>
              <h1 className="text-2xl font-bold tracking-tight">About</h1>
              <span className="mt-1 inline-flex items-center gap-1.5 rounded-md bg-muted px-2 py-1 text-xs font-mono">
                <Tag className="h-3.5 w-3.5" />
                {formatVersion(version)}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">What is {PRODUCT_NAME}</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            {PRODUCT_NAME} is WSP's camera trap analysis app. It detects
            animals, people and vehicles with MegaDetector and identifies
            species with SpeciesNet and WSP's own models, which are shared
            through the WSP model library on OneDrive. Your images and
            results stay on your computer.
          </p>
        </section>

        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">Support</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Questions, problems and ideas:{" "}
            <a href={ISSUES_URL} className="text-primary hover:underline">
              {ISSUES_URL.replace("https://", "")}
            </a>
          </p>

          {contributors && contributors.length > 0 && (
            <div className="mt-4 rounded-lg border bg-zinc-50 p-4">
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
                Code contributors
              </div>
              <div className="flex flex-wrap gap-2">
                {contributors.map((c) => (
                  <a
                    key={c.id}
                    href={c.html_url}
                    title={`${c.login} · ${c.contributions} commit${c.contributions === 1 ? "" : "s"}`}
                    className="block rounded-full ring-2 ring-transparent hover:ring-primary transition-colors"
                  >
                    <img
                      src={c.avatar_url}
                      alt={c.login}
                      className="h-9 w-9 rounded-full"
                      loading="lazy"
                      // Hide the avatar (and its anchor) instead of
                      // showing a broken-image icon if the GitHub
                      // CDN is unreachable.
                      onError={(e) => {
                        const a = e.currentTarget.closest("a");
                        if (a) a.style.display = "none";
                      }}
                    />
                  </a>
                ))}
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                Pulled live from GitHub. Order is by commit count, not
                contribution size.
              </p>
            </div>
          )}
        </section>

        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">Source and license</h2>
          <div className="mt-2 text-sm text-muted-foreground space-y-2">
            <div>
              Source code:{" "}
              <a href={PROJECT_URL} className="text-primary hover:underline">
                github.com/{REPO}
              </a>
            </div>
            <div>
              License:{" "}
              <a href={LICENSE_URL} className="text-primary hover:underline">
                MIT
              </a>
              . Third-party open-source components and their notices are
              listed in the LICENSE file installed with the app.
            </div>
            <p>
              The detection and classification models come from various
              developers (MegaDetector: Dan Morris; SpeciesNet: Google).
              Each has its own license, citation and intended use; see the{" "}
              <span className="font-medium">Model details</span> link below
              each model in the project settings.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
