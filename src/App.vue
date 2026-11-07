<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import SynodosMark from '@/components/SynodosMark.vue'

const route = useRoute()
const sidebarCollapsed = ref(false)
const mobileNavigationOpen = ref(false)
const isMeeting = computed(() => route.name === 'meeting')
watch(() => route.fullPath, () => { mobileNavigationOpen.value = false })
</script>

<template>
  <div class="app-shell workspace-shell synodos-shell" :class="{
    'sidebar-collapsed': sidebarCollapsed,
    'meeting-route': isMeeting,
    'mobile-sidebar-open': mobileNavigationOpen,
  }">
    <button v-if="mobileNavigationOpen" type="button" class="workspace-mobile-scrim"
      aria-label="Close navigation" @click="mobileNavigationOpen = false"></button>

    <header class="synodos-topbar">
      <div class="synodos-topbar-branding">
        <RouterLink to="/" class="workspace-brand synodos-brand" title="Synodos · Home">
          <SynodosMark class="synodos-brand-mark" :size="52" />
          <span class="synodos-brand-name">Synodos</span>
        </RouterLink>
        <div v-if="isMeeting" id="synodos-global-context" class="synodos-global-context"
          aria-label="Current meeting"></div>
      </div>
      <div class="synodos-topbar-actions">
        <div v-if="isMeeting" id="synodos-global-actions" class="synodos-global-actions"
          aria-label="Meeting controls"></div>
        <span v-else class="synodos-local-status"><span aria-hidden="true"></span> Local</span>
      </div>
      <button type="button" class="workspace-mobile-menu synodos-mobile-menu"
        aria-label="Open navigation" @click="mobileNavigationOpen = true">
        <AppIcon name="list" :size="22" />
      </button>
    </header>

    <aside class="workspace-sidebar" aria-label="Workspace navigation">
      <nav class="workspace-nav" aria-label="Main navigation">
        <RouterLink to="/" class="workspace-nav-item"
          :class="{ active: route.name === 'home' && route.hash !== '#meetings' }"
          title="Dashboard">
          <AppIcon name="home" :size="19" /><span>Dashboard</span>
        </RouterLink>
        <RouterLink to="/#meetings" class="workspace-nav-item"
          :class="{ active: isMeeting || route.hash === '#meetings' }"
          title="Meetings">
          <AppIcon name="file-text" :size="19" /><span>Meetings</span>
          <span v-if="isMeeting" class="sidebar-current-indicator" aria-hidden="true"></span>
        </RouterLink>
        <RouterLink to="/speakers" class="workspace-nav-item"
          :class="{ active: ['speakers', 'speaker-new'].includes(route.name as string) }"
          title="Speakers">
          <AppIcon name="users" :size="19" /><span>Speakers</span>
        </RouterLink>
        <div class="synodos-sidebar-divider" aria-hidden="true"></div>
        <RouterLink to="/hotwords" class="workspace-nav-item"
          :class="{ active: ['hotwords', 'hotword-new'].includes(route.name as string) }"
          title="Terminology">
          <AppIcon name="flame" :size="19" /><span>Terminology</span>
        </RouterLink>
        <RouterLink to="/settings" class="workspace-nav-item"
          :class="{ active: route.name === 'settings' }"
          title="Settings">
          <AppIcon name="settings" :size="19" /><span>Settings</span>
        </RouterLink>
      </nav>
      <div class="sidebar-bottom">
        <button type="button" class="workspace-collapse-btn"
          :aria-label="sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
          :aria-expanded="!sidebarCollapsed" @click="sidebarCollapsed = !sidebarCollapsed">
          <AppIcon name="list" :size="17" />
          <span>{{ sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation' }}</span>
        </button>
      </div>
    </aside>

    <div class="workspace-main-column">
      <div class="workspace-page-content">
        <RouterView />
      </div>
    </div>
  </div>
</template>
