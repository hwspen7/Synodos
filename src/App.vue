<script setup lang="ts">
import { useRoute } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'

const route = useRoute()
</script>

<template>
  <div class="app-shell">
    <header class="app-bar">
      <RouterLink to="/" class="brand">
        <span class="brand-mark">M</span>
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
