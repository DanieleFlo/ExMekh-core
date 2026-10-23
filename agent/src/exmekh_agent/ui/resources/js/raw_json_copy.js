async (event) => {
    const button = event.target.closest('.raw-json-copy-image');
    if (!button) return;
    event.stopPropagation();
    try {
        await navigator.clipboard.writeText(button.dataset.imageData);
        emit({ copied: true });
    } catch {
        emit({ copied: false });
    }
}
