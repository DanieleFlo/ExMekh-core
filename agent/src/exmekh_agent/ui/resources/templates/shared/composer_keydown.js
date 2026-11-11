(event) => {
    const textarea = event.target;
    if (!(textarea instanceof HTMLTextAreaElement) || textarea.readOnly || event.isComposing) return;

    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (!event.repeat) emit();
        return;
    }

    if (event.key !== 'F4' || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    if (event.repeat) return;

    const text = 'basito';
    // Preserve the browser's undo history when insertText is available.
    if (!document.execCommand('insertText', false, text)) {
        textarea.setRangeText(text, textarea.selectionStart, textarea.selectionEnd, 'end');
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
    }
}
