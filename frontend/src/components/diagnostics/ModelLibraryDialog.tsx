/**
 * WSP: pick the WSP model library folder.
 *
 * The library is the OneDrive (SharePoint) folder models are installed
 * from, usually "<OneDrive>/.../WSP CameraTrap/models". The app finds it by
 * itself in the usual OneDrive locations; this dialog is for when it
 * cannot, or when the library lives on a network share. Saving re-reads
 * the library's catalog, so newly published models appear straight away.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderOpen, Library } from "lucide-react";
import { toast } from "sonner";
import { wspApi } from "../../api/wsp";
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
  settings: "chosen in this dialog",
  autodetect: "found in OneDrive automatically",
};

export function ModelLibraryDialog({ open, onOpenChange }: ModelLibraryDialogProps) {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  const { data: status, isLoading } = useQuery({
    queryKey: ["wsp-library"],
    queryFn: wspApi.getLibrary,
    enabled: open,
  });

  const save = useMutation({
    mutationFn: (dir: string | null) => wspApi.setLibrary(dir),
    onSuccess: (next) => {
      setError(null);
      queryClient.setQueryData(["wsp-library"], next);
      // Newly published models show up in every model picker.
      void queryClient.invalidateQueries();
      toast.success(
        next.library_dir
          ? "Model library connected"
          : "No model library found. Models already installed keep working.",
      );
    },
    onError: (err: Error) => setError(err.message),
  });

  const chooseFolder = async () => {
    const dir = await window.electronAPI?.selectFolder?.();
    if (dir) save.mutate(dir);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Library className="h-5 w-5" />
            WSP model library
          </DialogTitle>
          <DialogDescription>
            Models are installed from the shared "WSP CameraTrap" folder on
            OneDrive. Sync it (Add shortcut to My files) and the app finds it
            by itself; otherwise pick its <span className="font-medium">models</span>{" "}
            folder here.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 text-sm">
          <div className="rounded-md border bg-muted/40 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Library in use
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
              <div className="mt-1 text-amber-700">
                {status?.configured_dir
                  ? `The saved folder cannot be reached: ${status.configured_dir}. Is OneDrive synced?`
                  : "No library found."}
              </div>
            )}
          </div>
          {status && (
            <p className="text-xs text-muted-foreground">
              Installed models are kept in{" "}
              <span className="break-all font-mono">{status.models_dir}</span>. A model
              folder copied there by hand works too.
            </p>
          )}
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        <DialogFooter className="gap-2 sm:justify-between">
          <Button
            variant="ghost"
            disabled={save.isPending || !status?.configured_dir}
            onClick={() => save.mutate(null)}
          >
            Find automatically
          </Button>
          <Button disabled={save.isPending} onClick={() => void chooseFolder()}>
            <FolderOpen className="mr-2 h-4 w-4" />
            {save.isPending ? "Connecting…" : "Choose folder…"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
