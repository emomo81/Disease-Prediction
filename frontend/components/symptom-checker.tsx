'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { AlertCircle, CheckCircle2, Loader2, Search, X } from 'lucide-react'
import api, { type Symptom, type Prediction } from '@/lib/api'

export function SymptomChecker() {
  const [symptoms, setSymptoms] = useState<Symptom[]>([])
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([])
  const [predictions, setPredictions] = useState<Prediction[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingSymptoms, setLoadingSymptoms] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    loadSymptoms()
  }, [])

  async function loadSymptoms() {
    try {
      setLoadingSymptoms(true)
      const data = await api.getSymptoms()
      setSymptoms(data.symptoms)
    } catch (err) {
      setError('Failed to load symptoms. Please ensure the Flask backend is running.')
    } finally {
      setLoadingSymptoms(false)
    }
  }

  const filteredSymptoms = symptoms.filter((symptom) =>
    symptom.label.toLowerCase().includes(searchQuery.toLowerCase())
  )

  function toggleSymptom(symptomValue: string) {
    setSelectedSymptoms((prev) =>
      prev.includes(symptomValue)
        ? prev.filter((s) => s !== symptomValue)
        : [...prev, symptomValue]
    )
    setPredictions(null)
    setError(null)
  }

  function clearSymptoms() {
    setSelectedSymptoms([])
    setPredictions(null)
    setError(null)
  }

  async function handlePredict() {
    if (selectedSymptoms.length === 0) {
      setError('Please select at least one symptom')
      return
    }

    try {
      setLoading(true)
      setError(null)
      const data = await api.predictDisease(selectedSymptoms)
      setPredictions(data.predictions)
    } catch (err: any) {
      setError(err.response?.data?.error || 'Prediction failed. Please try again.')
      setPredictions(null)
    } finally {
      setLoading(false)
    }
  }

  if (loadingSymptoms) {
    return (
      <div className="min-h-screen py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <Skeleton className="h-12 w-3/4 mx-auto mb-4" />
          <Skeleton className="h-6 w-1/2 mx-auto mb-12" />
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {Array.from({ length: 12 }).map((_, i) => (
              <Skeleton key={i} className="h-12" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen py-24 px-6 bg-gradient-to-b from-background to-muted/20">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Disease Prediction System
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Select your symptoms below to get AI-powered disease predictions with confidence scores
          </p>
        </div>

        {/* Search Bar */}
        <Card className="mb-8">
          <CardContent className="pt-6">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search symptoms..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </CardContent>
        </Card>

        {/* Selected Symptoms */}
        {selectedSymptoms.length > 0 && (
          <Card className="mb-8 bg-primary/5 border-primary/20">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">
                  Selected Symptoms ({selectedSymptoms.length})
                </CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearSymptoms}
                  className="h-8"
                >
                  <X className="h-4 w-4 mr-1" />
                  Clear All
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {selectedSymptoms.map((symptomValue) => {
                  const symptom = symptoms.find((s) => s.value === symptomValue)
                  return (
                    <Badge
                      key={symptomValue}
                      variant="secondary"
                      className="px-3 py-1 cursor-pointer hover:bg-destructive hover:text-destructive-foreground transition-colors"
                      onClick={() => toggleSymptom(symptomValue)}
                    >
                      {symptom?.label}
                      <X className="ml-2 h-3 w-3" />
                    </Badge>
                  )
                })}
              </div>
            </CardContent>
            <CardFooter>
              <Button
                onClick={handlePredict}
                disabled={loading}
                size="lg"
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="mr-2 h-5 w-5" />
                    Predict Disease
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>
        )}

        {/* Error Message */}
        {error && (
          <Card className="mb-8 border-destructive bg-destructive/10">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3 text-destructive">
                <AlertCircle className="h-5 w-5 flex-shrink-0" />
                <p>{error}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Prediction Results */}
        {predictions && predictions.length > 0 && (
          <div className="mb-12">
            <h2 className="text-2xl font-bold mb-6">Prediction Results</h2>
            <div className="grid gap-6 md:grid-cols-3">
              {predictions.map((prediction, index) => (
                <Card
                  key={index}
                  className={`transition-all hover:shadow-lg ${
                    index === 0 ? 'border-primary shadow-md' : ''
                  }`}
                >
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="text-sm text-muted-foreground mb-1">
                          {index === 0 ? 'Most Likely' : `Alternative ${index}`}
                        </div>
                        <CardTitle className="text-xl">
                          {prediction.disease}
                        </CardTitle>
                      </div>
                      <Badge
                        variant={index === 0 ? 'default' : 'secondary'}
                        className="text-lg px-3 py-1"
                      >
                        {prediction.confidence.toFixed(1)}%
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {prediction.description}
                    </p>
                  </CardContent>
                  <CardFooter>
                    <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full ${
                          index === 0
                            ? 'bg-primary'
                            : 'bg-muted-foreground'
                        } transition-all duration-500`}
                        style={{ width: `${prediction.confidence}%` }}
                      />
                    </div>
                  </CardFooter>
                </Card>
              ))}
            </div>
            <div className="mt-6 text-center">
              <Button variant="outline" onClick={clearSymptoms}>
                Start New Diagnosis
              </Button>
            </div>
          </div>
        )}

        {/* Symptoms Grid */}
        {!predictions && (
          <>
            <h2 className="text-2xl font-bold mb-6">
              {searchQuery ? 'Search Results' : 'All Symptoms'}
            </h2>
            {filteredSymptoms.length === 0 ? (
              <Card>
                <CardContent className="py-12 text-center text-muted-foreground">
                  No symptoms found matching &quot;{searchQuery}&quot;
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {filteredSymptoms.map((symptom) => {
                  const isSelected = selectedSymptoms.includes(symptom.value)
                  return (
                    <button
                      key={symptom.value}
                      onClick={() => toggleSymptom(symptom.value)}
                      className={`px-4 py-3 rounded-lg border-2 text-left transition-all hover:scale-105 ${
                        isSelected
                          ? 'border-primary bg-primary text-primary-foreground font-medium'
                          : 'border-border bg-card hover:border-primary/50'
                      }`}
                    >
                      {symptom.label}
                    </button>
                  )
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
