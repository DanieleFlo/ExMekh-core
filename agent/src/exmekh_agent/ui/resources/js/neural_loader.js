// Decorative data flow; its tokens never come from conversation content.
const HEIGHT = 34;
const SPEED = 2.5;
const LAYERS = [[0], [-0.65, 0.65], [-0.85, 0.85], [0], [-0.85, 0.85], [-0.65, 0.65], [0]];
const COLORS = [[86, 182, 206], [158, 142, 212], [216, 124, 156]];
const choose = values => values[Math.floor(Math.random() * values.length)];
const rgba = (color, alpha) => `rgba(${color.join(',')},${alpha})`;
const digit = () => String(Math.floor(Math.random() * 19) - 9);

function layerColor(index) {
    const start = index <= 3 ? COLORS[0] : COLORS[1];
    const end = index <= 3 ? COLORS[1] : COLORS[2];
    const fraction = index <= 3 ? index / 3 : (index - 3) / 3;
    return start.map((value, channel) => Math.round(value + (end[channel] - value) * fraction));
}

class NeuralAnimation {
    constructor(canvas, tokens) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.tokens = tokens;
        this.active = false;
        this.width = 0;
        this.time = 0;
        this.frame = null;
        this.nodes = [];
        this.edges = [];
        this.particles = [];
        this.streams = [];
        this.nextToken = [0, 0.5];
        this.motion = window.matchMedia('(prefers-reduced-motion: reduce)');
        this.onVisibility = () => this.sync();
        this.onMotion = () => this.sync();
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(canvas);
        document.addEventListener('visibilitychange', this.onVisibility);
        this.motion.addEventListener('change', this.onMotion);
        this.resize();
    }

    resize() {
        this.width = this.canvas.getBoundingClientRect().width;
        if (!this.width || !this.ctx) {
            this.stop();
            return;
        }
        const scale = window.devicePixelRatio || 1;
        this.canvas.width = Math.round(this.width * scale);
        this.canvas.height = Math.round(HEIGHT * scale);
        this.ctx.setTransform(scale, 0, 0, scale, 0, 0);
        this.nodes = [];
        this.edges = [];
        const margin = this.width * 0.14;
        const layers = LAYERS.map((offsets, index) => offsets.map(offset => {
            const node = {
                x: margin + index * (this.width - margin * 2) / 6,
                baseY: HEIGHT / 2 + offset * 10,
                y: HEIGHT / 2 + offset * 10,
                color: layerColor(index),
                radius: index === 3 ? 2.6 : 1.8,
                phase: Math.random() * Math.PI * 2,
                pulse: 0,
                outgoing: [],
            };
            this.nodes.push(node);
            return node;
        }));
        layers.slice(0, -1).forEach((layer, index) => {
            layer.forEach(from => layers[index + 1].forEach(to => {
                const gradient = this.ctx.createLinearGradient(from.x, 0, to.x, 0);
                gradient.addColorStop(0, rgba(from.color, 1));
                gradient.addColorStop(1, rgba(to.color, 1));
                const edge = {from, to, gradient};
                from.outgoing.push(edge);
                this.edges.push(edge);
            }));
        });
        const particleCount = Math.max(10, Math.min(24, Math.round(this.width / 34)));
        this.particles = Array.from({length: particleCount}, () => ({
            edge: choose(this.edges),
            progress: Math.random(),
            speed: 0.14 + Math.random() * 0.14,
            text: digit(),
            changeAt: this.time + Math.random(),
        }));
        this.resetStreams();
        this.sync();
    }

    resetStreams() {
        this.streams = this.tokens.length ? [
            {side: 0, progress: 0.35, text: choose(this.tokens), speed: 0.18},
            {side: 1, progress: 0.25, text: choose(this.tokens), speed: 0.17},
        ] : [];
        this.nextToken = [this.time + 0.5, this.time + 1];
    }

    setTokens(tokens) {
        this.tokens = tokens;
        this.resetStreams();
        if (this.active && this.width) this.draw();
    }

    setActive(active) {
        this.active = active;
        this.sync();
    }

    stop() {
        if (this.frame !== null) cancelAnimationFrame(this.frame);
        this.frame = null;
    }

    sync() {
        this.stop();
        if (!this.active || !this.width || document.hidden || !this.ctx) return;
        this.draw();
        if (this.motion.matches) return;
        this.previousTime = null;
        this.frame = requestAnimationFrame(now => this.tick(now));
    }

    tick(now) {
        const delta = this.previousTime === null ? 0
            : Math.min((now - this.previousTime) / 1000, 0.05) * SPEED;
        this.previousTime = now;
        this.time += delta;
        this.nodes.forEach(node => {
            node.y = node.baseY + Math.sin(this.time * 0.6 + node.phase) * 0.6;
            node.pulse = Math.max(0, node.pulse - delta * 1.5);
        });
        this.particles.forEach(particle => {
            particle.progress += delta * particle.speed;
            if (this.time >= particle.changeAt) {
                particle.text = digit();
                particle.changeAt = this.time + 0.3 + Math.random() * 0.4;
            }
            if (particle.progress >= 1) {
                const target = particle.edge.to;
                target.pulse = 0.6;
                particle.edge = choose(target.outgoing.length && Math.random() > 0.18
                    ? target.outgoing : this.nodes[0].outgoing);
                particle.progress = 0;
            }
        });
        this.streams.forEach(stream => {
            stream.progress += delta * stream.speed;
            if (stream.side === 0 && stream.progress >= 1) this.nodes[0].pulse = 0.55;
        });
        this.streams = this.streams.filter(stream => stream.progress < 1);
        if (this.tokens.length) {
            [0, 1].forEach(side => {
                const inTransit = this.streams.filter(stream => stream.side === side);
                const distance = Math.min(58, this.width * 0.11);
                const hasSpace = inTransit.every(stream => stream.progress * distance
                    > this.ctx.measureText(stream.text).width + 12);
                if (this.time >= this.nextToken[side] && inTransit.length < 2 && hasSpace) {
                    this.streams.push({side, progress: 0, text: choose(this.tokens), speed: 0.14 + Math.random() * 0.06});
                    this.nextToken[side] = this.time + 0.8 + Math.random();
                }
            });
        }
        this.draw();
        this.frame = requestAnimationFrame(next => this.tick(next));
    }

    controls(edge) {
        const dx = (edge.to.x - edge.from.x) * 0.45;
        return [edge.from.x + dx, edge.from.y * 0.72 + HEIGHT * 0.14,
            edge.to.x - dx, edge.to.y * 0.72 + HEIGHT * 0.14];
    }

    position(edge, progress) {
        const [x1, y1, x2, y2] = this.controls(edge);
        const inverse = 1 - progress;
        const curve = (start, first, second, end) => inverse ** 3 * start
            + 3 * inverse ** 2 * progress * first + 3 * inverse * progress ** 2 * second + progress ** 3 * end;
        return [curve(edge.from.x, x1, x2, edge.to.x), curve(edge.from.y, y1, y2, edge.to.y)];
    }

    waveIntensity(x) {
        return Math.max(0, Math.sin(x / this.width * 3.8 - this.time * 0.65));
    }

    draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, HEIGHT);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const distance = Math.min(58, this.width * 0.11);
        const first = this.nodes[0];
        const last = this.nodes[this.nodes.length - 1];
        ctx.setLineDash([2, 4]);
        [first, last].forEach((node, side) => {
            const start = side ? node.x : node.x - distance;
            const end = side ? node.x + distance : node.x;
            const gradient = ctx.createLinearGradient(start, 0, end, 0);
            gradient.addColorStop(0, rgba(node.color, side ? 0.25 : 0));
            gradient.addColorStop(1, rgba(node.color, side ? 0 : 0.25));
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 0.7;
            ctx.beginPath();
            ctx.moveTo(start, node.y);
            ctx.lineTo(end, node.y);
            ctx.stroke();
        });
        ctx.setLineDash([]);
        this.edges.forEach(edge => {
            const wave = this.waveIntensity(edge.from.x);
            ctx.globalAlpha = 0.14 + wave * 0.36;
            ctx.strokeStyle = edge.gradient;
            ctx.lineWidth = 0.65 + wave * 0.25;
            ctx.beginPath();
            ctx.moveTo(edge.from.x, edge.from.y);
            ctx.bezierCurveTo(...this.controls(edge), edge.to.x, edge.to.y);
            ctx.stroke();
        });
        ctx.globalAlpha = 1;
        ctx.font = '500 8.5px "SF Mono", Consolas, monospace';
        this.particles.forEach(particle => {
            const [x, y] = this.position(particle.edge, particle.progress);
            const fade = Math.min(1, particle.progress / 0.15, (1 - particle.progress) / 0.15);
            const opacity = fade * (0.28 + this.waveIntensity(x) * 0.72);
            ctx.fillStyle = `rgba(226,232,240,${opacity * 0.88})`;
            ctx.fillText(particle.text, x, y);
        });
        ctx.font = '600 8px "SF Mono", Consolas, monospace';
        this.streams.forEach(stream => {
            const node = stream.side ? last : first;
            const x = stream.side ? node.x + stream.progress * distance : node.x - (1 - stream.progress) * distance;
            const fade = stream.side ? Math.min(stream.progress / 0.15, (1 - stream.progress) / 0.85)
                : Math.min(1, stream.progress / 0.25);
            const opacity = fade * (0.28 + this.waveIntensity(x) * 0.72);
            ctx.fillStyle = stream.side ? `rgba(248,225,235,${opacity * 0.92})` : `rgba(215,238,245,${opacity * 0.92})`;
            ctx.fillText(stream.text, x, node.y, Math.min(48, this.width * 0.11));
        });
        this.nodes.forEach(node => {
            const radius = node.radius * (1 + node.pulse * 0.25);
            const halo = radius * 2.6;
            const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, halo);
            glow.addColorStop(0, rgba(node.color, 0.4 + node.pulse * 0.15));
            glow.addColorStop(1, rgba(node.color, 0));
            ctx.fillStyle = glow;
            ctx.beginPath();
            ctx.arc(node.x, node.y, halo, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = rgba(node.color, 1);
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = 'rgba(255,255,255,0.75)';
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius * 0.4, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    destroy() {
        this.stop();
        this.resizeObserver.disconnect();
        document.removeEventListener('visibilitychange', this.onVisibility);
        this.motion.removeEventListener('change', this.onMotion);
    }
}

export default {
    template: '<canvas ref="canvas" class="input-neural-loader" role="status" :aria-label="label" />',
    props: {active: Boolean, tokens: Array, label: String},
    mounted() {
        this.animation = new NeuralAnimation(this.$refs.canvas, this.tokens);
        this.animation.setActive(this.active);
    },
    watch: {
        active(value) { this.$nextTick(() => this.animation.setActive(value)); },
        tokens(value) { this.animation.setTokens(value); },
    },
    beforeUnmount() { this.animation.destroy(); },
};
