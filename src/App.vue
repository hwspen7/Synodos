<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'

const route = useRoute()
const sidebarCollapsed = ref(false)
const mobileNavigationOpen = ref(false)
const isMeeting = computed(() => route.name === 'meeting')
const sectionTitle = computed(() => {
  if (route.name === 'home') return 'Workspace'
  if (route.name === 'meeting') return 'Meeting room'
  if (route.name === 'speakers' || route.name === 'speaker-new') return 'Speaker library'
  if (route.name === 'hotwords' || route.name === 'hotword-new') return 'Terminology'
  return 'Workspace'
})
watch(() => route.fullPath, () => { mobileNavigationOpen.value = false })
</script>

<template>
  <div class="app-shell workspace-shell" :class="{ 'sidebar-collapsed': sidebarCollapsed, 'meeting-route': isMeeting, 'mobile-sidebar-open': mobileNavigationOpen }">
    <button
      v-if="mobileNavigationOpen"
      class="workspace-mobile-scrim"
      type="button"
      aria-label="Close navigation"
      @click="mobileNavigationOpen = false"
    ></button>
    <aside class="workspace-sidebar" aria-label="Workspace navigation">
      <RouterLink to="/" class="workspace-brand" title="Meeting of Pioneers">
        <span class="workspace-brand-symbol" aria-hidden="true">
          <svg viewBox="0 0 32 32" width="25" height="25" fill="none">
            <path d="M27.5 5 4.5 14.3l9.2 3.1L27.5 5Z" fill="currentColor"/>
            <path d="m27.5 5-9.6 23-4.2-10.6L27.5 5Z" fill="currentColor" opacity=".78"/>
            <path d="m13.7 17.4-1.4 7 5.6-4.4" stroke="currentColor" stroke-width="1.3"/>
          </svg>
        </span>
        <span class="workspace-brand-copy"><strong>Meeting</strong><small>of Pioneers</small></span>
      </RouterLink>

      <div class="sidebar-divider"></div>
      <div class="workspace-nav-label">WORKSPACE</div>
      <nav class="workspace-nav" aria-label="Main navigation">
        <RouterLink to="/" class="workspace-nav-item" :class="{ active: route.name === 'home' && route.hash !== '#meetings' }" title="Overview">
          <AppIcon name="home" :size="19" /><span>Overview</span>
        </RouterLink>
        <RouterLink to="/#meetings" class="workspace-nav-item" :class="{ active: route.name === 'meeting' || route.hash === '#meetings' }" title="Meetings">
          <AppIcon name="file-text" :size="19" /><span>Meetings</span>
          <span v-if="isMeeting" class="sidebar-current-indicator" aria-hidden="true"></span>
        </RouterLink>
        <RouterLink to="/speakers" class="workspace-nav-item" :class="{ active: ['speakers', 'speaker-new'].includes(route.name as string) }" title="Speakers">
          <AppIcon name="users" :size="19" /><span>Speakers</span>
        </RouterLink>
        <RouterLink to="/hotwords" class="workspace-nav-item" :class="{ active: ['hotwords', 'hotword-new'].includes(route.name as string) }" title="Terminology">
          <AppIcon name="flame" :size="19" /><span>Terminology</span>
        </RouterLink>
      </nav>

      <div class="sidebar-bottom">
        <div class="sidebar-help-card">
          <span class="sidebar-help-mark"><AppIcon name="mic" :size="17" /></span>
          <strong>Private by design</strong>
          <p>Your recordings and transcripts remain in your local workspace.</p>
        </div>
        <button
          class="workspace-collapse-btn"
          type="button"
          :aria-label="sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
          :aria-expanded="!sidebarCollapsed"
          @click="sidebarCollapsed = !sidebarCollapsed"
        >
          <AppIcon name="list" :size="17" /><span>{{ sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation' }}</span>
        </button>
      </div>
    </aside>

    <div class="workspace-main-column">
      <header class="workspace-topbar">
        <div class="workspace-topbar-left">
          <button type="button" class="workspace-mobile-menu" aria-label="Open navigation" @click="mobileNavigationOpen = true"><AppIcon name="list" :size="20" /></button>
          <span class="workspace-location">WORKSPACE <span aria-hidden="true">/</span> <strong>{{ sectionTitle }}</strong></span>
        </div>
        <div class="workspace-topbar-right">
          <span class="workspace-local-pill"><span aria-hidden="true"></span> Local workspace</span>
          <RouterLink to="/#meetings" class="workspace-quick-action"><AppIcon name="file-text" :size="17" /> Browse meetings</RouterLink>
        </div>
      </header>
      <div class="workspace-page-content"><RouterView /></div>
    </div>
  </div>
</template>
