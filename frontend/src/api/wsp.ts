/**
 * The WSP model library setting (/api/wsp/library).
 */

import { api } from "../lib/api-client";

export interface LibraryStatus {
  /** The library folder in use, or null when none is reachable. */
  library_dir: string | null;
  /** "env", "settings", "link", "autodetect", or null. */
  source: string | null;
  /** The folder saved in the app, even when it is unreachable now. */
  configured_dir: string | null;
  /** The OneDrive/SharePoint share link in use, or null. */
  library_url: string | null;
  models_dir: string;
  download_in_progress: boolean;
  download_progress: number;
  download_message: string;
  download_error: string | null;
}

export const wspApi = {
  getLibrary: () => api.get<LibraryStatus>("/api/wsp/library"),
  /** Use a folder. */
  setLibraryDir: (libraryDir: string) =>
    api.post<LibraryStatus>("/api/wsp/library", { library_dir: libraryDir }),
  /** Use a share link; the download runs in the background. */
  setLibraryUrl: (libraryUrl: string) =>
    api.post<LibraryStatus>("/api/wsp/library", { library_url: libraryUrl }),
  /** Forget the saved folder and link: back to the shipped link / OneDrive. */
  resetLibrary: () => api.post<LibraryStatus>("/api/wsp/library", {}),
  /** Check the link again (downloads only when the file changed). */
  refreshLibrary: () => api.post<LibraryStatus>("/api/wsp/library/refresh", {}),
};
