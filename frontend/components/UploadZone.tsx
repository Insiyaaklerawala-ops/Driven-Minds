"use client";

import { useState, useCallback, DragEvent } from "react";
import { UploadCloud, FileCheck2 } from "lucide-react";
import clsx from "clsx";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  fileName: string | null;
  isUploading: boolean;
}

export function UploadZone({ onFileSelect, fileName, isUploading }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith(".csv")) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={clsx(
        "border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer",
        isDragging ? "border-signal bg-signal/5" : "border-border hover:border-ink-faint",
        isUploading && "opacity-60 pointer-events-none"
      )}
    >
      <input
        type="file"
        accept=".csv"
        id="csv-upload"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelect(file);
        }}
      />
      <label htmlFor="csv-upload" className="cursor-pointer flex flex-col items-center gap-2">
        {fileName ? (
          <>
            <FileCheck2 className="w-6 h-6 text-signal" />
            <span className="text-sm font-mono text-ink-primary">{fileName}</span>
            <span className="text-xs text-ink-faint">click or drop to replace</span>
          </>
        ) : (
          <>
            <UploadCloud className="w-6 h-6 text-ink-muted" />
            <span className="text-sm text-ink-primary">
              {isUploading ? "Uploading..." : "Drop CSV or click to browse"}
            </span>
            <span className="text-xs text-ink-faint">.csv files only</span>
          </>
        )}
      </label>
    </div>
  );
}