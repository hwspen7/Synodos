import './assets/main.css'
import './assets/workspace-system.css'
import './assets/synodos-theme.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

createApp(App).use(router).mount('#app')
