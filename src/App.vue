<script setup lang="ts">
import { useRoute } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'

const route = useRoute()
</script>

<template>
  <div class="app-shell">
    <header class="app-bar">
      <RouterLink to="/" class="brand">
        <span class="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 3 3 10.2l7.4 2.3L21 3Z" />
            <path d="m21 3-7.5 18-3.1-8.5L21 3Z" />
            <path d="m10.4 12.5-1 6.2 4.1-3.2" />
          </svg>
        </span>
        <span class="brand-name">Meeting of Pioneers</span>
      </RouterLink>
      <nav class="app-nav">
        <RouterLink to="/" class="nav-link" :class="{ active: route.name === 'home' }">
          <AppIcon name="home" :size="15" />
          Home
        </RouterLink>
        <RouterLink
          to="/speakers"
          class="nav-link"
          :class="{ active: ['speakers', 'speaker-new'].includes(route.name as string) }"
        >
          <AppIcon name="users" :size="15" />
          Speakers
        </RouterLink>
        <RouterLink
          to="/hotwords"
          class="nav-link"
          :class="{ active: ['hotwords', 'hotword-new'].includes(route.name as string) }"
        >
          <AppIcon name="flame" :size="15" />
          Hotwords
        </RouterLink>
      </nav>
    </header>
    <RouterView />
  </div>
</template>
