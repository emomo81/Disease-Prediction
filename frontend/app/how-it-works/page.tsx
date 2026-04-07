import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { CheckCircle } from 'lucide-react'

export const metadata = {
  title: 'How It Works | Disease Prediction',
  description: 'Learn how our AI disease prediction system works',
}

const steps = [
  {
    number: '01',
    title: 'Select Your Symptoms',
    description:
      'Browse through our comprehensive list of symptoms or use the search function to quickly find what you\'re experiencing. Select all symptoms that apply to your current condition.',
  },
  {
    number: '02',
    title: 'AI Analysis',
    description:
      'Our advanced machine learning models analyze your symptom combination against a database of 820+ diseases. Multiple algorithms work together to ensure accurate predictions.',
  },
  {
    number: '03',
    title: 'Get Predictions',
    description:
      'Receive instant predictions with confidence scores. We show you the top 3 most likely conditions based on your symptoms, complete with descriptions and probability percentages.',
  },
]

const techDetails = [
  'Random Forest with 100-200 trees for robust classification',
  'XGBoost for gradient boosting optimization',
  'Extra Trees for variance reduction',
  'Cross-validation for model reliability',
  'Automated hyperparameter tuning',
  'Regular model updates with new data',
]

export default function HowItWorksPage() {
  return (
    <div className="min-h-screen">
      {/* Header */}
      <div className="bg-gradient-to-b from-primary/10 to-background py-24 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            How It Works
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Understanding the technology behind our AI-powered disease prediction system
          </p>
        </div>
      </div>

      {/* Process Steps */}
      <div className="py-24 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="space-y-12">
            {steps.map((step, index) => (
              <Card
                key={index}
                className="border-2 hover:border-primary/50 transition-colors"
              >
                <CardHeader>
                  <div className="flex items-start gap-6">
                    <div className="text-5xl font-bold text-primary/20">
                      {step.number}
                    </div>
                    <div className="flex-1">
                      <CardTitle className="text-2xl mb-2">{step.title}</CardTitle>
                      <CardDescription className="text-base">
                        {step.description}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* Technology Section */}
      <div className="py-24 px-6 bg-muted/30">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">The Technology</h2>
            <p className="text-lg text-muted-foreground">
              Our system employs multiple machine learning algorithms for maximum accuracy
            </p>
          </div>

          <Card>
            <CardContent className="pt-6">
              <ul className="grid md:grid-cols-2 gap-4">
                {techDetails.map((detail, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <CheckCircle className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span>{detail}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          {/* Disclaimer */}
          <Card className="mt-12 bg-yellow-500/10 border-yellow-500/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span className="text-2xl">⚠️</span>
                Important Disclaimer
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                This system is designed for educational and informational purposes only.
                It should not be used as a substitute for professional medical advice,
                diagnosis, or treatment. Always seek the advice of your physician or other
                qualified health provider with any questions you may have regarding a medical
                condition.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* CTA */}
      <div className="py-24 px-6 text-center">
        <div className="max-w-2xl mx-auto">
          <h2 className="text-3xl font-bold mb-6">Ready to Try It?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Experience the power of AI-driven disease prediction
          </p>
          <div className="flex gap-4 justify-center">
            <Button asChild size="lg">
              <Link href="/app">Start Diagnosis</Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/">Back to Home</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
