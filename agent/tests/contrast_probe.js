root => {
    const ctx = document.createElement('canvas').getContext('2d', {willReadFrequently: true});
    const colors = new Map();
    const styles = new Map();
    const transparent = [0, 0, 0, 0];
    const white = [255, 255, 255, 1];
    const findings = [];
    let checked = 0;

    function rgba(value) {
        if (!colors.has(value)) {
            ctx.clearRect(0, 0, 1, 1);
            ctx.fillStyle = 'transparent';
            ctx.fillStyle = value;
            ctx.fillRect(0, 0, 1, 1);
            const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
            colors.set(value, [r, g, b, a / 255]);
        }
        return colors.get(value);
    }

    function css(element) {
        if (!styles.has(element)) styles.set(element, getComputedStyle(element));
        return styles.get(element);
    }

    function over(top, bottom) {
        const alpha = top[3] + bottom[3] * (1 - top[3]);
        if (!alpha) return transparent;
        return [0, 1, 2].map(i => (
            top[i] * top[3] + bottom[i] * bottom[3] * (1 - top[3])
        ) / alpha).concat(alpha);
    }

    function luminance(rgb) {
        const linear = rgb.slice(0, 3).map(value => {
            const channel = value / 255;
            return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
        });
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
    }

    function visible(element) {
        if (!element.getClientRects().length || css(element).visibility !== 'visible' ||
            element.closest('script, style, [inert], [disabled], [aria-disabled=true], .q-field--disabled, .disabled')) return false;
        const drawer = element.closest('.q-drawer');
        if (!drawer) return true;
        const {left, right} = drawer.getBoundingClientRect();
        return right > 0 && left < window.innerWidth;
    }

    function measure(element, label, color, minimum, opacity = 1, ownSurface = true) {
        const foreground = rgba(color).slice();
        foreground[3] *= opacity;
        if (!foreground[3]) return;
        let background = transparent;
        let ink = foreground;
        let complex = false;
        for (let node = ownSurface ? element : element.parentElement; node; node = node.parentElement) {
            const style = css(node);
            const surface = rgba(style.backgroundColor);
            if (style.backgroundImage !== 'none' && background[3] < 1) complex = true;
            const shadowOnly = /^drop-shadow\([^)]*\)$/.test(style.filter.replace(/rgba?\([^)]*\)/g, ''));
            if ((style.filter !== 'none' && !shadowOnly) || style.mixBlendMode !== 'normal') complex = true;
            background = over(background, surface).slice();
            ink = over(ink, surface).slice();
            background[3] *= Number(style.opacity);
            ink[3] *= Number(style.opacity);
            if (!Number(style.opacity)) return;
        }
        background = over(background, white);
        ink = over(ink, white);
        const levels = [luminance(background), luminance(ink)].sort((a, b) => b - a);
        const ratio = (levels[0] + 0.05) / (levels[1] + 0.05);
        const selector = [element.localName, ...element.classList].join('.');
        if (complex || element.closest('.image-generation-progress')) {
            findings.push({status: 'WARNING', label, selector, reason: 'Background or compositing requires visual review', minimum});
        } else {
            checked++;
            if (ratio < minimum) findings.push({
                status: 'NOT PASS', label, selector, minimum, ratio: Number(ratio.toFixed(3)),
                foreground: ink.slice(0, 3).map(Math.round), background: background.slice(0, 3).map(Math.round),
            });
        }
    }

    function minimumTextContrast(style) {
        const size = parseFloat(style.fontSize);
        return size >= 24 || (size >= 56 / 3 && Number(style.fontWeight) >= 700) ? 3 : 4.5;
    }

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const label = node.textContent.trim().slice(0, 120);
        const element = node.parentElement;
        if (!label || !visible(element)) continue;
        const style = css(element);
        const icon = style.fontFamily.includes('Material Icons');
        if (icon && !element.closest('button, [role=button], [role=tab], [role=combobox]')) continue;
        const svgText = element.closest('text');
        measure(element, label, svgText ? style.fill : style.color, icon ? 3 : minimumTextContrast(style));
    }

    for (const input of root.querySelectorAll('input:not([type=hidden]), textarea')) {
        if (!visible(input)) continue;
        const style = input.value ? css(input) : getComputedStyle(input, '::placeholder');
        const label = input.type === 'password' ? 'Password field' : input.value || input.placeholder;
        if (label) measure(input, label.slice(0, 120), style.color, minimumTextContrast(style), input.value ? 1 : Number(style.opacity));
    }

    for (const control of root.querySelectorAll('.q-field--outlined .q-field__control, .theme-switch, .exmekh-chat-input-box')) {
        if (!visible(control)) continue;
        const outlined = control.matches('.q-field__control');
        const border = outlined ? getComputedStyle(control, '::before') : css(control);
        const label = control.closest('.q-field')?.querySelector('.q-field__label')?.textContent ||
            control.getAttribute('aria-label') || (outlined ? 'Input' : 'Chat composer');
        measure(control, `${label} outline`, border.borderTopColor, 3, Number(border.opacity), false);
    }

    for (const icon of root.querySelectorAll('.theme-switch svg')) {
        if (visible(icon)) measure(icon, `${icon.classList[0]} outline`, css(icon).stroke, 3);
    }

    return {checked, findings};
}
