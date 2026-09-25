import * as React from "react";

interface CheckboxProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  indeterminate?: boolean;
  className?: string;
}

export function Checkbox({ checked, onCheckedChange, indeterminate, className }: CheckboxProps) {
  const ref = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (ref.current) {
      ref.current.indeterminate = indeterminate || false;
    }
  }, [indeterminate]);

  // WSP red colors: #E02F28 (primary - full), #F4A19D (lighter - half/indeterminate)
  const checkboxStyle = indeterminate
    ? { accentColor: '#F4A19D' }
    : { accentColor: '#E02F28' };

  return (
    <input
      ref={ref}
      type="checkbox"
      checked={checked}
      onChange={(e) => onCheckedChange(e.target.checked)}
      style={checkboxStyle}
      className={`h-4 w-4 shrink-0 rounded border-gray-300 focus:ring-2 focus:ring-[#E02F28] focus:ring-offset-2 ${className || ""}`}
    />
  );
}
