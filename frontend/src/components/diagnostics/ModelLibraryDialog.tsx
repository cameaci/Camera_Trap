/**
 * Install models: from a model library .zip, or from a library folder.
 *
 * The .zip (built with wsp/tools/wsp_library.py bundle, and shared however
 * WSP likes, e.g. on OneDrive) is unpacked straight into the local models
 * folder. A library folder (a synced OneDrive folder or a network share)
 * is where the app copies models from on demand. Either way the catalog
 * is re-read, so the models appear straight away.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileArchive, FolderOpen, Library, Loader2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { wspApi, type LibraryStatus } from "../../api/wsp";
import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

interface ModelLibraryDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const SOURCE_LABELS: Record<string, string> = {
  env: "set by WSP_MODEL_LIBRARY_DIR",
  settings: "folder chosen in this dialog",
  autodetect: "synced OneDrive folder, found automatically",
};

export function ModelLibraryDialog({ open, onOpenChange }: ModelLibraryDialogProps) {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  const { data: status, isLoading } = useQuery({
    queryKey: ["wsp-library"],
    queryFn: wspApi.getLibrary,
    enabled: open,
  });

  const afterChange = (next: LibraryStatus) => {
    queryClient.setQueryData(["wsp-library"], next);
    // Every model picker needs the new catalog.
    void queryClient.invalidateQueries();
  };

  const setFolder = useMutation({
    mutationFn: (action: () => Promise<LibraryStatus>) => action(),
    onMutate: () => setError(null),
    onSuccess: (next) => {
      afterChange(next);
      toast.success(next.library_dir ? "Model library folder connected" : "Library folder removed");
    },
    onError: (err: Error) => setError(err.message),
  });

  const importZip = useMutation({
    mutationFn: (zipPath: string) => wspApi.importZip(zipPath),
    onMutate: () => setError(null),
    onSuccess: (result) => {
      afterChange(result.status);
      toast.success(
        result.imported.length
          ? `Installed ${result.imported.length} model${result.imported.length === 1 ? "" : "s"}`
          : "The .zip held no model folders; its catalog was updated",
      );
    },
    onError: (err: Error) => setError(err.message),
  });

  const chooseZip = async () => {
    const path = await window.electronAPI?.openFile?.({
      title: "Choose the WSP model library .zip",
      filters: [{ name: "Model library", extensions: ["zip"] }],
    });
    if (path) importZip.mutate(path);
  };

  const chooseFolder = async () => {
    const dir = await window.electronAPI?.selectFolder?.();
    if (dir) setFolder.mutate(() => wspApi.setLibraryDir(dir));
  };

  const busy = setFolder.isPending || importZip.isPending;

  return (
    <Dialog open={open} onOpenChange={(next) => !importZip.isPending && onOpenChange(next)}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Library className="h-5 w-5" />
            WSP model library
          </DialogTitle>
          <DialogDescription>
            Install models from the WSP model library .zip
            (WSP-CameraTrap-models.zip), or use a folder that holds the library,
            such as a synced OneDrive folder.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 text-sm">
          <div className="space-y-2">
            <Button
              className="w-full"
              disabled={busy}
              onClick={() => void chooseZip()}
            >
              {importZip.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <FileArchive className="mr-2 h-4 w-4" />
              )}
              {importZip.isPending ? "Installing models…" : "Install models from a .zip…"}
            </Button>
            {status && (
              <p className="text-xs text-muted-foreground">
                The models are unpacked into{" "}
                <span className="break-all font-mono">{status.models_dir}</span>. This
                takes a minute for a large .zip.
              </p>
            )}
          </div>

          <div className="rounded-md border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Library folder
            </div>
            {isLoading ? (
              <div className="mt-1 text-muted-foreground">Checking…</div>
            ) : status?.library_dir ? (
              <>
                <div className="mt-1 break-all font-mono text-xs">{status.library_dir}</div>
                {status.source && (
                  <div className="mt-1 text-xs text-muted-foreground">
                    {SOURCE_LABELS[status.source] ?? status.source}
                  </div>
                )}
              </>
            ) : (
              <div className="mt-1 text-muted-foreground">
                {status?.configured_dir
                  ? `The saved folder cannot be reached: ${status.configured_dir}. Is OneDrive synced?`
                  : "None. Not needed when models are installed from a .zip."}
              </div>
            )}
          </div>

          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        <DialogFooter className="gap-2 sm:justify-between">
          <Button
            variant="ghost"
            disabled={busy || !status?.configured_dir}
            onClick={() => setFolder.mutate(() => wspApi.resetLibrary())}
          >
            Remove folder
          </Button>
          <Button variant="outline" disabled={busy} onClick={() => void chooseFolder()}>
            <FolderOpen className="mr-2 h-4 w-4" />
            Use a folder…
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
