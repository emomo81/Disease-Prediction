'use client'
import React from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { InfiniteSlider } from '@/components/ui/infinite-slider'
import { ProgressiveBlur } from '@/components/ui/progressive-blur'
import { ThemeToggle } from '@/components/theme-toggle'
import { cn } from '@/lib/utils'
import { Menu, X, ChevronRight, Activity } from 'lucide-react'
import { useScroll, motion } from 'framer-motion'

export function HeroSection() {
    return (
        <>
            <HeroHeader />
            <main className="overflow-x-hidden">
                <section>
                    <div className="">
                        <div className="aspect-2/3 relative z-10 flex flex-col justify-end px-6 lg:aspect-video">
                            <div className="mx-auto w-full max-w-7xl pb-6 lg:px-12 lg:pb-32">
                                <div className="max-w-lg">
                                    <h1 className="text-balance text-5xl md:text-6xl xl:text-7xl">
                                        AI-Powered Disease Prediction with 95-99% Accuracy
                                    </h1>
                                    <p className="mt-6 text-balance text-lg">
                                        Advanced machine learning analyzes your symptoms across 820+ diseases to provide instant, accurate health predictions.
                                    </p>

                                    <div className="mt-8 flex items-center gap-2">
                                        <Button
                                            asChild
                                            size="lg"
                                            className="h-12 rounded-full pl-5 pr-3 text-base">
                                            <Link href="/app">
                                                <span className="text-nowrap">Start Diagnosis</span>
                                                <ChevronRight className="ml-1" />
                                            </Link>
                                        </Button>
                                        <Button
                                            key={2}
                                            asChild
                                            size="lg"
                                            variant="ghost"
                                            className="h-12 rounded-full px-5 text-base hover:bg-zinc-950/5 dark:hover:bg-white/5">
                                            <Link href="/features">
                                                <span className="text-nowrap">View Features</span>
                                            </Link>
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="aspect-2/3 pointer-events-none absolute inset-1 overflow-hidden rounded-3xl border border-black/10 lg:aspect-video lg:rounded-[3rem] dark:border-white/5">
                            <video
                                autoPlay
                                loop
                                muted
                                playsInline
                                className="not-dark:invert size-full -scale-x-100 object-cover"
                                src="https://videos.pexels.com/video-files/35968183/15249566_1920_1080_30fps.mp4"
                            />
                        </div>
                    </div>
                </section>
                <section className="bg-background pb-2">
                    <div className="group relative m-auto max-w-7xl px-6">
                        <div className="flex flex-col items-center md:flex-row">
                            <div className="md:max-w-44 md:border-r md:pr-6">
                                <p className="text-end text-sm">Trusted Healthcare Analytics</p>
                            </div>
                            <div className="relative py-6 md:w-[calc(100%-11rem)]">
                                <InfiniteSlider
                                    speedOnHover={20}
                                    speed={40}
                                    gap={112}>
                                    <div className="flex items-center gap-3">
                                        <div className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">820+</div>
                                        <div className="text-sm text-muted-foreground">Diseases</div>
                                    </div>

                                    <div className="flex items-center gap-3">
                                        <div className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">95-99%</div>
                                        <div className="text-sm text-muted-foreground">Accuracy</div>
                                    </div>

                                    <div className="flex items-center gap-3">
                                        <div className="text-4xl font-bold bg-gradient-to-r from-pink-600 to-red-600 bg-clip-text text-transparent">10K+</div>
                                        <div className="text-sm text-muted-foreground">Symptoms</div>
                                    </div>

                                    <div className="flex items-center gap-3">
                                        <div className="text-4xl font-bold bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent">Instant</div>
                                        <div className="text-sm text-muted-foreground">Results</div>
                                    </div>

                                    <div className="flex items-center gap-3">
                                        <div className="text-4xl font-bold bg-gradient-to-r from-orange-600 to-yellow-600 bg-clip-text text-transparent">ML</div>
                                        <div className="text-sm text-muted-foreground">Powered</div>
                                    </div>

                                    <div className="flex items-center gap-3">
                                        <div className="text-4xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">24/7</div>
                                        <div className="text-sm text-muted-foreground">Available</div>
                                    </div>
                                </InfiniteSlider>

                                <div className="bg-linear-to-r from-background absolute inset-y-0 left-0 w-20"></div>
                                <div className="bg-linear-to-l from-background absolute inset-y-0 right-0 w-20"></div>
                                <ProgressiveBlur
                                    className="pointer-events-none absolute left-0 top-0 h-full w-20"
                                    direction="left"
                                    blurIntensity={1}
                                />
                                <ProgressiveBlur
                                    className="pointer-events-none absolute right-0 top-0 h-full w-20"
                                    direction="right"
                                    blurIntensity={1}
                                />
                            </div>
                        </div>
                    </div>
                </section>
            </main>
        </>
    )
}

const menuItems = [
    { name: 'Features', href: '/features' },
    { name: 'How It Works', href: '/how-it-works' },
    { name: 'About', href: '/about' },
]

const HeroHeader = () => {
    const [menuState, setMenuState] = React.useState(false)
    const [scrolled, setScrolled] = React.useState(false)
    const { scrollYProgress } = useScroll()

    React.useEffect(() => {
        const unsubscribe = scrollYProgress.on('change', (latest) => {
            setScrolled(latest > 0.05)
        })
        return () => unsubscribe()
    }, [scrollYProgress])

    return (
        <header>
            <nav
                data-state={menuState && 'active'}
                className="group fixed z-20 w-full pt-2">
                <div className={cn('mx-auto max-w-7xl rounded-3xl px-6 transition-all duration-300 lg:px-12', scrolled && 'bg-background/50 backdrop-blur-2xl')}>
                    <motion.div
                        key={1}
                        className={cn('relative flex flex-wrap items-center justify-between gap-6 py-3 duration-200 lg:gap-0 lg:py-6', scrolled && 'lg:py-4')}>
                        <div className="flex w-full items-center justify-between gap-12 lg:w-auto">
                            <Link
                                href="/"
                                aria-label="home"
                                className="flex items-center space-x-2">
                                <Logo />
                            </Link>

                            <button
                                onClick={() => setMenuState(!menuState)}
                                aria-label={menuState == true ? 'Close Menu' : 'Open Menu'}
                                className="relative z-20 -m-2.5 -mr-4 block cursor-pointer p-2.5 lg:hidden">
                                <Menu className="group-data-[state=active]:rotate-180 group-data-[state=active]:scale-0 group-data-[state=active]:opacity-0 m-auto size-6 duration-200" />
                                <X className="group-data-[state=active]:rotate-0 group-data-[state=active]:scale-100 group-data-[state=active]:opacity-100 absolute inset-0 m-auto size-6 -rotate-180 scale-0 opacity-0 duration-200" />
                            </button>

                            <div className="hidden lg:block">
                                <ul className="flex gap-8 text-sm">
                                    {menuItems.map((item, index) => (
                                        <li key={index}>
                                            <Link
                                                href={item.href}
                                                className="text-muted-foreground hover:text-accent-foreground block duration-150">
                                                <span>{item.name}</span>
                                            </Link>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>

                        <div className="bg-background group-data-[state=active]:block lg:group-data-[state=active]:flex mb-6 hidden w-full flex-wrap items-center justify-end space-y-8 rounded-3xl border p-6 shadow-2xl shadow-zinc-300/20 md:flex-nowrap lg:m-0 lg:flex lg:w-fit lg:gap-6 lg:space-y-0 lg:border-transparent lg:bg-transparent lg:p-0 lg:shadow-none dark:shadow-none dark:lg:bg-transparent">
                            <div className="lg:hidden">
                                <ul className="space-y-6 text-base">
                                    {menuItems.map((item, index) => (
                                        <li key={index}>
                                            <Link
                                                href={item.href}
                                                className="text-muted-foreground hover:text-accent-foreground block duration-150">
                                                <span>{item.name}</span>
                                            </Link>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            <div className="flex w-full flex-col space-y-3 sm:flex-row sm:gap-3 sm:space-y-0 md:w-fit">
                                <ThemeToggle />
                                <Button
                                    asChild
                                    size="sm">
                                    <Link href="/app">
                                        <span>Start Diagnosis</span>
                                    </Link>
                                </Button>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </nav>
        </header>
    )
}

const Logo = ({ className }: { className?: string }) => {
    return (
        <div className="flex items-center gap-2">
            <Activity className={cn('h-6 w-6 text-primary', className)} />
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                HealthAI
            </span>
        </div>
    )
}
