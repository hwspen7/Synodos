<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import SynodosMark from '@/components/SynodosMark.vue'

const route = useRoute()
const sidebarCollapsed = ref(false)
const mobileNavigationOpen = ref(false)
const isMeeting = computed(() => route.name === 'meeting')
const meetingHref = computed(() => isMeeting.value
  ? '/meeting/' + encodeURIComponent(String(route.params.id))
  : '/#meetings')
const summaryHref = computed(() => isMeeting.value
  ? meetingHref.value + '?tool=review'
  : '/#meetings')
const settingsHref = computed(() => isMeeting.value
  ? meetingHref.value + '?tool=settings'
  : '/settings')
const sectionTitle = computed(() => {
  if (route.name === 'home') return 'Dashboard'
  if (route.name === 'meeting') return 'Meetings'
  if (route.name === 'speakers' || route.name === 'speaker-new') return 'Speakers'
  if (route.name === 'hotwords' || route.name === 'hotword-new') return 'Terminology'
  if (route.name === 'settings') return 'Settings'
  return 'Dashboard'
})
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
        <span class="synodos-brand-divider" aria-hidden="true"></span>
        <span class="synodos-tagline">Meet <b>·</b> Transcribe <b>·</b> Understand</span>
      </div>
      <div class="synodos-topbar-actions">
        <span class="synodos-local-status"><span aria-hidden="true"></span> Local workspace</span>
        <RouterLink to="/#meetings" class="synodos-topbar-link"><AppIcon name="file-text" :size="17" /> Meetings</RouterLink>
        <span class="synodos-profile-bubble" title="Synodos local workspace" aria-label="Local workspace">S</span>
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
        <RouterLink :to="meetingHref" class="workspace-nav-item"
          :class="{ active: isMeeting && route.query.tool !== 'review' && route.query.tool !== 'settings' }"
          title="Live Captions">
          <AppIcon name="mic" :size="19" /><span>Live Captions</span>
        </RouterLink>
        <RouterLink to="/speakers" class="workspace-nav-item"
          :class="{ active: ['speakers', 'speaker-new'].includes(route.name as string) }"
          title="Speakers">
          <AppIcon name="users" :size="19" /><span>Speakers</span>
        </RouterLink>
        <RouterLink :to="summaryHref" class="workspace-nav-item"
          :class="{ active: isMeeting && route.query.tool === 'review' }"
          title="Summaries">
          <AppIcon name="file-text" :size="19" /><span>Summaries</span>
        </RouterLink>
        <div class="synodos-sidebar-divider" aria-hidden="true"></div>
        <RouterLink to="/hotwords" class="workspace-nav-item"
          :class="{ active: ['hotwords', 'hotword-new'].includes(route.name as string) }"
          title="Terminology">
          <AppIcon name="flame" :size="19" /><span>Terminology</span>
        </RouterLink>
        <RouterLink :to="settingsHref" class="workspace-nav-item"
          :class="{ active: route.name === 'settings' || (isMeeting && route.query.tool === 'settings') }"
          title="Settings">
          <AppIcon name="settings" :size="19" /><span>Settings</span>
        </RouterLink>
      </nav>
      <div class="sidebar-bottom">
        <div class="synodos-privacy-note">
          <span class="synodos-privacy-symbol"><AppIcon name="check" :size="15" /></span>
          <strong>Private by design</strong>
          <p>Transcripts and recordings stay in your workspace.</p>
        </div>
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
