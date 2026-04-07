import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity, Brain, Clock, Database, Lock, Zap } from 'lucide-react'

export const metadata = {
  title: 'Features | Disease Prediction',
  description: 'Explore the powerful features of our AI disease prediction system',
}

const features = [
  {
    icon: Brain,
    title: 'Advanced Machine Learning',
    description: 'Our system uses state-of-the-art algorithms including Random Forest, XGBoost, and Extra Trees to analyze symptoms and predict diseases with exceptional accuracy.',
  },
  {
    icon: Database,
    title: '820+ Diseases Covered',
    description: 'Comprehensive disease database covering a wide range of conditions from common ailments to rare diseases, ensuring thorough diagnostic coverage.',
  },
  {
    icon: Zap,
    title: '95-99% Accuracy',
    description: 'Trained on verified medical datasets with rigorous testing, our models achieve industry-leading accuracy rates for disease prediction.',
  },
  {
    icon: Clock,
    title: 'Instant Results',
    description: 'Get predictions in seconds. Our optimized system processes symptoms and returns top predictions with confidence scores immediately.',
  },
  {
    icon: Activity,
    title: 'Confidence Scoring',
    description: 'Each prediction includes a detailed confidence score, helping you understand the likelihood of each potential diagnosis.',
  },
  {
    icon: Lock,
    title: 'Privacy First',
    description: 'Your health data is never stored. All predictions are computed in real-time without retaining personal information.',
  },
]

export default function FeaturesPage() {
  return (
    <div className="min-h-screen">
      {/* Header */}
      <div className="bg-gradient-to-b from-primary/10 to-background py-24 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Powerful Features for Accurate Predictions
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Our AI-powered system combines cutting-edge technology with medical expertise
            to deliver reliable disease predictions.
          </p>
        </div>
      </div>

      {/* Features Grid */}
      <div className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <Card key={index} className="border-2 hover:border-primary/50 transition-colors">
                  <CardHeader>
                    <div className="mb-4 p-3 bg-primary/10 rounded-lg w-fit">
                      <Icon className="h-6 w-6 text-primary" />
                    </div>
                    <CardTitle className="text-xl">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-base">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {/* CTA Section */}
          <div className="mt-24 text-center">
            <Card className="bg-gradient-to-r from-primary/10 to-purple-500/10 border-primary/20">
              <CardContent className="py-12">
                <h2 className="text-3xl font-bold mb-4">
                  Ready to try it yourself?
                </h2>
                <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
                  Start using our disease prediction system now. No registration required.
                </p>
                <Button asChild size="lg" className="h-12 px-8 text-base">
                  <Link href="/app">Start Diagnosis</Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Back Link */}
      <div className="pb-12 px-6 text-center">
        <Link href="/" className="text-primary hover:underline">
          ← Back to Home
        </Link>
      </div>
    </div>
  )
}
