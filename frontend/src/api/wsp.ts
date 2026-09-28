/**
 * The WSP model library setting (/api/wsp/library).
 */

import { api } from "../lib/api-client";

export interface LibraryStatus {
  /** The library folder in use, or null when none is reachable. */
  library_dir: string | null;
  /** "env", "settings", "autodetect", or null. */
  source: string | null;
  /** The folder saved in the app, even when it is unreachable now. */
  configured_dir: string | null;
  models_dir: string;
}

export interface ZipImportResult {
  /** The installed model folders, as "<type>/<id>". */
  imported: string[];
  status: LibraryStatus;
}

export const wspApi = {
  getLibrary: () => api.get<LibraryStatus>("/api/wsp/library"),
  /** Use a library folder. */
  setLibraryDir: (libraryDir: string) =>
    api.post<LibraryStatus>("/api/wsp/library", { library_dir: libraryDir }),
  /** Forget the saved library folder. */
  resetLibrary: () => api.post<LibraryStatus>("/api/wsp/library", {}),
  /** Unpack a model library .zip into the local models folder. */
  importZip: (zipPath: string) =>
    api.post<ZipImportResult>("/api/wsp/library/import", { zip_path: zipPath }),
};
