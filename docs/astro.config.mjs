import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
//import starlightThemeObsidian from 'starlight-theme-obsidian'
import starlightThemeRapide from 'starlight-theme-rapide'

// https://astro.build/config
export default defineConfig({
	site: 'https://demiotic.github.io',
	base: '/reminiscence',
	integrations: [
		starlight({
			title: 'Reminiscence',
			//plugins: [starlightThemeObsidian()],
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
					label: 'Production',
					items: [
						{ label: 'OpenTelemetry', link: '/production/opentelemetry' },
						{ label: 'Health Checks', link: '/production/health-checks' },
						{ label: 'Performance', link: '/production/performance' },
						{ label: 'Best Practices', link: '/production/best-practices' },
					],
				},
				{
					label: 'Examples',
					items: [
						{ label: 'LLM Applications', link: '/examples/llm-apps' },
						{ label: 'RAG Pipelines', link: '/examples/rag' },
						{ label: 'Multi-Agent Systems', link: '/examples/multi-agent' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'API Documentation', link: '/reference/api' },
						{ label: 'Configuration Options', link: '/reference/config' },
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
