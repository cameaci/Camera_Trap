/**
 * WSP: the model library folder setting (/api/wsp/library).
 */

import { api } from "../lib/api-client";

export interface LibraryStatus {
  /** The library in use, or null when none is reachable. */
  library_dir: string | null;
  /** "env", "settings", "autodetect", or null. */
  source: string | null;
  /** The folder saved in Settings, even when it is unreachable now. */
  configured_dir: string | null;
  models_dir: string;
}

export const wspApi = {
  getLibrary: () => api.get<LibraryStatus>("/api/wsp/library"),
  /** Save a folder (null goes back to finding the OneDrive folder automatically). */
  setLibrary: (libraryDir: string | null) =>
    api.post<LibraryStatus>("/api/wsp/library", { library_dir: libraryDir }),
};
