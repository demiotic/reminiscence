import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
//import starlightThemeObsidian from 'starlight-theme-obsidian'
import starlightThemeRapide from 'starlight-theme-rapide'
import d2 from 'astro-d2'

// https://astro.build/config
export default defineConfig({
	site: 'https://demiotic.github.io',
	base: '/',
	integrations: [
		d2({
			d2: '/home/midnattsol/.local/bin/d2',
		}),
		starlight({
			title: 'Reminiscence',
			plugins: [starlightThemeRapide()],
			description: 'Semantic caching for LLMs - reduce costs and latency through intelligent response caching',
			logo: {
				src: './src/assets/logo.svg',
			},
			social: [
				{
					label: 'GitHub',
					icon: 'github',
					href: 'https://github.com/demiotic/reminiscence',
				},
			],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', link: '/' },
						{ label: 'Installation', link: '/getting-started/installation' },
						{ label: 'Quick Start', link: '/getting-started/quick-start' },
					],
				},
				{
					label: 'Core Concepts',
					items: [
						{ label: 'How It Works', link: '/concepts/how-it-works' },
						{ label: 'Semantic Matching', link: '/concepts/semantic-matching' },
						{ label: 'Hybrid Caching', link: '/concepts/hybrid-caching' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Basic Usage', link: '/guides/basic-usage' },
						{ label: 'Configuration', link: '/guides/configuration' },
						{ label: 'Decorators', link: '/guides/decorators' },
						{ label: 'Background Tasks', link: '/guides/background-tasks' },
					],
				},
				{
					label: 'API Reference',
					items: [
						{ label: 'API Overview', link: '/reference/api' },
						{ label: 'Core Operations', link: '/api/core-operations' },
						{ label: 'Decorators', link: '/api/decorators' },
						{ label: 'Configuration', link: '/api/configuration' },
						{ label: 'Data Types', link: '/api/data-types' },
					],
				},
				{
					label: 'Dual-Plane Architecture',
					items: [
						{ label: 'Overview', link: '/dual-plane/overview' },
						{ label: 'Control Plane (gRPC)', link: '/dual-plane/grpc-api' },
						{ label: 'Data Plane (Flight)', link: '/dual-plane/flight-dataplane' },
						{ label: 'Server Configuration', link: '/dual-plane/grpc-server' },
					],
				},
				{
					label: 'Examples',
					items: [
						{ label: 'LLM Applications', link: '/examples/llm-apps' },
						{ label: 'RAG Pipelines', link: '/examples/rag' },
						{ label: 'Multi-Agent Systems', link: '/examples/multi-agent' },
						{ label: 'gRPC Microservices', link: '/examples/grpc-microservices' },
					],
				},
				{
					label: 'Production',
					items: [
						{ label: 'OpenTelemetry', link: '/production/opentelemetry' },
						{ label: 'Health Checks', link: '/production/health-checks' },
						{ label: 'Performance', link: '/production/performance' },
						{ label: 'Best Practices', link: '/production/best-practices' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Configuration Options', link: '/reference/config' },
						{ label: 'Data Types & Serialization', link: '/reference/data-types' },
						{ label: 'Storage Architecture', link: '/reference/storage-architecture' },
						{ label: 'Metrics', link: '/reference/metrics' },
					],
				},
			],
			customCss: [
				'./src/styles/custom.css',
			],
		}),
	],
});
